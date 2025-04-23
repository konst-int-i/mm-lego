from mm_lego.pipeline import PipelineBase
from mm_lego.utils import setup_logging, Config
import pandas as pd
from mm_lego.models import HealNet
from mm_lego.models.train_tasks import SurvivalTrainer, ClassificationTrainer
from mmhb.loader import TCGASurvivalDataset, MimicDataset, ISICDataset
from mm_lego.models import LegoBlock, LegoFuse, LegoMerge, SNN, MILAttentionNet, Ensemble
from pathlib import Path
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import *
from box import Box
import random
from timeit import timeit
import wandb
from mmhb.loader import MimicDataset
from torch.utils.data import Subset
from mm_lego.pipeline import Fusion




logger = logger= setup_logging()


def run_ablation(config: Box, overlap: float, folds, model: str = "legomerge"):

    # folds = config.folds

    c_indeces = []

    for fold in range(1, folds+1):

        logger.info(f"*****Fold {fold}*****")
        torch.manual_seed(fold)
        np.random.seed(fold)
        torch.cuda.manual_seed(fold)

        wandb.init(mode="disabled")

        source = "mimic"
        config.data.source = source
        modalities = ["tab", "ts"]

        tab = MimicDataset(data_path=config.data.mimic.data_path, modalities=["tab"], dataset="icd9", expand=True)
        ts = MimicDataset(data_path=config.data.mimic.data_path, modalities=["ts"], dataset="icd9", expand=True)
        # shared
        mm = MimicDataset(data_path=config.data.mimic.data_path, modalities=["tab", "ts"], dataset="icd9", expand=True)
        tab, ts, tab_indeces, ts_indeces = _create_subsample(tab, ts, N=10000, f=overlap)
        # get mm indeces that don't overlap with ts or tab ones
        mm_indeces = list(set(range(len(mm))) - set(tab_indeces) - set(ts_indeces))
        mm = SubsetDataset(mm, mm_indeces)

        # print overlapping set between tab and ts indices
        print(f"Intersection: {len(set(tab.indices).intersection(ts.indices))}")
        print(f"Disjoint samples/symmetric difference (per set): {len(set(tab.indices).symmetric_difference(ts.indices))/2}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        total_size = len(tab)
        train_size = int(0.7 * total_size)
        test_size = int(0.15 * total_size)
        val_size = total_size - train_size - test_size


        # working from here
        generator = torch.Generator().manual_seed(fold)

        tab_train, tab_val, tab_test = torch.utils.data.random_split(tab, [train_size, val_size, test_size], generator=generator)
        ts_train, ts_val, ts_test = torch.utils.data.random_split(ts, [train_size, val_size, test_size], generator=generator)

        mm_test_indices = torch.randperm(len(mm), generator=generator)[:val_size]
        mm_test = SubsetDataset(mm, mm_test_indices)



        tab_weights = torch.Tensor(_calc_class_weights(tab_train, config)).to(device)
        ts_weights = torch.Tensor(_calc_class_weights(ts_train, config)).to(device)

        workers = 8
        datasets = {
            "tab": [tab_train, tab_val, tab_test],
            "ts": [ts_train, ts_val, ts_test],
            "mm": [mm_test],
        }

        loaders = {}

        for key, dataset_list in datasets.items():
            loaders[key] = [
                DataLoader(dataset, num_workers=workers, **config.loader.to_dict())
                for dataset in dataset_list
            ]

        # Access loaders like this
        tab_train_loader, tab_val_loader, tab_test_loader = loaders["tab"]
        ts_train_loader, ts_val_loader, ts_test_loader = loaders["ts"]
        mm_test_loader = loaders["mm"][0]


        # define models
        if model == "legomerge":
            config.model = model
            tab_block = LegoBlock(
                in_shape=next(iter(tab_train_loader))[0][0].shape[1:],
                num_classes=len(torch.unique(tab.targets)),
                **config.model_params.lego.to_dict()
            ).to(device)
            ts_block = LegoBlock(
                in_shape=next(iter(ts_train_loader))[0][0].shape[1:],
                num_classes=len(torch.unique(ts.targets)),
                **config.model_params.lego.to_dict()
            ).to(device)
        elif model == "ensemble":
            config.model = model
            tab_block = SNN(
                input_dim = next(iter(tab_train_loader))[0][0].shape[2],
                n_classes=len(torch.unique(tab.targets)),
                **config.model_params.snn.to_dict()
            ).to(device)
            ts_block = MILAttentionNet(
                input_dim = next(iter(ts_train_loader))[0][0].shape[1:],
                n_classes=len(torch.unique(ts.targets)),
                size_arg = source,
                **config.model_params.amil.to_dict()
            ).to(device)
            # tab_block = HealNet(
            #     modalities = 1,
            #     input_axes = [1],
            #     input_channels = [next(iter(tab_train_loader))[0][0].shape[2]],
            #     num_classes=len(torch.unique(tab.targets)),
            #     **config.model_params.healnet.to_dict(),
            # ).to(device)
            # ts_block = HealNet(
            #     modalities=1,
            #     input_axes=[1],
            #     input_channels = [next(iter(ts_train_loader))[0][0].shape[2]],
            #     num_classes=len(torch.unique(ts.targets)),
            #     **config.model_params.healnet.to_dict(),
            # ).to(device)


        tab_trainer = ClassificationTrainer(config, tab_block, tab_train_loader, tab_val_loader, tab_test_loader, tab_weights, fold)
        ts_trainer = ClassificationTrainer(config, ts_block, ts_train_loader, ts_val_loader, ts_test_loader, ts_weights, fold)

        for trainer in [tab_trainer, ts_trainer]:
            for epoch in range(1, config.epochs + 1):
                _, (train_c_index, val_c_index) = trainer.train_epoch(epoch)
                if trainer.stop:
                    break

        blocks = [tab_block, ts_block]


        # now merge
        if model == "legomerge":
            merge_model = LegoMerge(
                blocks=blocks,
                **config.model_params.lego.to_dict()
            )
            # merge_model = Ensemble(encoders=blocks)
        elif model == "ensemble":
            merge_model = Ensemble(encoders=blocks)
        merge_trainer = ClassificationTrainer(config, merge_model, tab_train_loader, tab_val_loader, mm_test_loader, tab_weights, fold)
        test_c_index, _ = merge_trainer.evaluate(epoch, mm_test_loader, subset="test")
        c_indeces.append(test_c_index)
    print(c_indeces)

    return config.model, np.mean(c_indeces), np.std(c_indeces)







def _get_subsample_indices(dataset, total_samples, exclude_indices=None):
    """ Helper function to get random subsample indices, excluding specific indices if provided. """
    all_indices = set(range(len(dataset)))
    if exclude_indices is not None:
        available_indices = list(all_indices - set(exclude_indices))
    else:
        available_indices = list(all_indices)
    return random.sample(available_indices, total_samples)

def _create_subsample(tab_dataset, ts_dataset, N, f):
    N_overlap = int(f * N)
    N_specific = int((1 - f) * N)

    # Get random indices for overlapping samples
    overlap_indices = _get_subsample_indices(tab_dataset, N_overlap)

    # Get random indices for specific samples, ensuring no overlap between them or with the overlap indices
    specific_indices_tab = _get_subsample_indices(tab_dataset, N_specific, exclude_indices=overlap_indices)
    specific_indices_ts = _get_subsample_indices(ts_dataset, N_specific, exclude_indices=overlap_indices + specific_indices_tab)

    # Combine indices
    tab_indices = overlap_indices + specific_indices_tab
    ts_indices = overlap_indices + specific_indices_ts

    # Create subsets using the combined indices
    tab_subset = SubsetDataset(tab_dataset, tab_indices)
    ts_subset = SubsetDataset(ts_dataset, ts_indices)

    return tab_subset, ts_subset, tab_indices, ts_indices

class SubsetDataset(Dataset):
    """A dataset that provides access to a subset of another dataset based on specified indices."""
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.original_dataset[original_idx]

    @property
    def targets(self):
        return torch.stack([self.original_dataset.targets[idx] for idx in self.indices])



def _calc_class_weights(train: Subset, config: Box) -> np.ndarray:

    if config.model_params.class_weights in ["inverse", "inverse_root"]:
        train_targets = np.array(train.dataset.targets)[train.indices]
        _, counts = np.unique(train_targets, return_counts=True)
        if config.model_params.class_weights == "inverse":
            class_weights = 1. / counts
        elif config.model_params.class_weights == "inverse_root":
            class_weights = 1. / np.sqrt(counts)
    else:
        class_weights = None
    return class_weights






if __name__ == "__main__":

    config = Config("config/config_dev.yml").read()

    folds = 1
    # overlaps = [1]
    overlaps = [0, 0.25, 0.5, 0.75, 1]
    # models = ["legomerge", "ensemble"]
    models = ["legomerge", "ensemble"]
    # result_df = pd.DataFrame(columns=["model", "overlap", "c_index"])
    result_df = pd.DataFrame()
    for model in models:
        for overlap in overlaps:
            model, mean_c_index, std_c_index = run_ablation(config, overlap=overlap, folds=folds, model=model)
            print(f"Overlap: {overlap}, c_index: {mean_c_index}, model: {model}")
            result_df = pd.concat([result_df, pd.DataFrame([{"model": model, "overlap": overlap, "symmetric difference": 1-overlap, "c_index_mean": mean_c_index, "c_index_std": std_c_index}])], ignore_index=True)

    print(result_df)
    result_df.to_csv("result_log/final/overlap_ablation.csv", index=False)