"""
Run fusion pipeline
"""
from mm_lego.pipeline import PipelineBase
from mm_lego.utils import setup_logging, Config
import pandas as pd
from mm_lego.models import HealNet
from mm_lego.models.train_tasks import SurvivalTrainer, ClassificationTrainer
from mmhb.loader import TCGASurvivalDataset, MimicDataset, ISICDataset
from mm_lego.models import LegoBlock, LegoFuse, LegoMerge
from pathlib import Path
import multiprocessing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import *
from box import Box
from timeit import timeit
import wandb

logger= setup_logging()

class Fusion(PipelineBase):

    def __init__(self, config: Box, **kwargs):

        super().__init__(config, **kwargs)

    def run_lego(self, blocks: List=None):

        models, train_c_indeces, val_c_indeces, test_c_indeces, t_epochs = [], [], [], [], []

        modalities = self.config.data[self.source].modalities


        for fold in range(1, self.config.folds + 1):
            logger.info(f"*****Fold {fold}*****")
            torch.manual_seed(fold)
            np.random.seed(fold)
            torch.cuda.manual_seed(fold)
            self.seed = fold


            self.blocks = []
            # load inside of loop for fixed seeds
            # ******** UNIMODAL *******
            for mod in modalities:
                # check if model has been cached
                if blocks is not None:
                    self.blocks = blocks
                    print("Using blocks passed in")
                    break
                cache_status, block = self._check_lego_cache(mod, fold)
                if cache_status:
                    self.blocks.append(block)
                    continue

                self.config.data[self.source].modalities = [mod]
                train_data, val_data, test_data = self.get_data()
                block = self.get_model(train_data)

                # check that this is an instance of legoblock
                assert isinstance(block, LegoBlock), "Unimodal must be an instance of LegoBlock"

                epoch, block, test_data, train_c_index, trainer, val_c_index = self.train_fold(fold, t_epochs, block, train_data, val_data, test_data)


                test_c_index, _ = trainer.evaluate(epoch, test_data, subset="test")
                # write model
                self._write_lego_cache(block, mod, fold)
                self.blocks.append(block)

            # ******** MULTIMODAL *******
            self.config.data[self.source].modalities = modalities
            tune_epochs = self.config.model_params.lego.tune_epochs
            train_data, val_data, test_data = self.get_data()
            model = self.get_model(train_data)
            wandb.watch(model)
            assert isinstance(model, (LegoFuse, LegoMerge)), f"Multimodal model must be an instance of LegoFuse/LegoMerge, got {type(model)}"

            if self.config.source in ["tcga"]:
                trainer = SurvivalTrainer(
                    self.config, model, train_data, val_data, test_data, self.class_weights, fold
                )
            elif self.config.source in ["mimic", "isic"]:
                trainer = ClassificationTrainer(self.config, model, train_data, val_data, test_data,
                                                self.class_weights, fold)
            # if isinstance(model, LegoFuse):
            if self.config.model in ["legofuse", "legomerge-tune"]:
                # fine-tuning only allowed for some models
                for epoch in range(1, tune_epochs + 1):
                    t_epoch, (train_c_index, val_c_index) = trainer.train_epoch(epoch)
                    t_epochs.append(t_epoch)
                    logger.info(f"Epoch time elapsed: {t_epoch:.3f} s")
                    if trainer.stop:
                        break
            else:
                epoch, train_c_index, val_c_index = None, 0., 0.
            test_c_index, _ = trainer.evaluate(epoch, test_data, subset="test")

            # keep track of models and indices for each fold
            models.append(model)
            train_c_indeces.append(train_c_index)
            val_c_indeces.append(val_c_index)
            test_c_indeces.append(test_c_index)

            wandb.log({"mean_train_c_index": np.mean(train_c_indeces),
                       "mean_val_c_index": np.mean(val_c_indeces),
                       "std_train_c_index": np.std(train_c_indeces),
                       "std_val_c_index": np.std(val_c_indeces),
                       "mean_test_c_index": np.mean(test_c_indeces),
                       "std_test_c_index": np.std(test_c_indeces),
                       "mean_train_epoch (s)": np.mean(t_epochs)})

            if self.config.write_model:
                if self.config.model == "legoblock":
                    write_path = Path(f"block_log/{self.dataset}/{wandb.run.name}/model_checkpoint_fold_{fold}.pth")
                else:
                    write_path = Path(f"model_log/{wandb.run.name}/model_checkpoint_fold_{fold}.pth")
                write_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model, write_path)

            # get best model across folds
        best_fold = np.argmax(test_c_indeces)
        best_model = models[best_fold]
        # save config
        if self.config.write_model:
            torch.save({"config": self.config, "best_fold": best_fold}, write_path.parent.joinpath("metadata.pth"))
            logger.info(f"Saved models and metadata (config, best fold) to {write_path.parent}")

        wandb.finish()

        return best_model, np.mean(test_c_indeces)


    def _check_lego_cache(self, mod: str, fold: int) -> Tuple[bool, nn.Module]:
        cache_path = Path("lego_cache/")
        # create dir
        cache_path.mkdir(parents=True, exist_ok=True)
        model_name = f"lego_{self.source}_{self.config.data[self.source].dataset}_{mod}_{fold}.pth"

        # check if model already exists
        if self.config.model_params.lego.use_cache:
            if self.config.model_params.lego.use_final:
                # load in unimodal block from final results
                result_df = pd.read_csv("result_log/results.csv")
                lookup = mod
                if mod == "ts":
                    lookup = "img"
                name = result_df[(result_df["dataset"] == self.dataset) & (result_df["model"] == "legoblock") & (
                            result_df["modalities"] == lookup)].name.values[0]
                model_path = Path(f"block_log/{self.dataset}/{name}/model_checkpoint_fold_{fold}.pth")
                print(f"Loading best LegoBlock from {model_path}")
                model = torch.load(model_path)
                return True, model
            else:
                if (cache_path / model_name).exists():
                    logger.info(f"Loading cached LegoBlock {model_name}")
                    model = torch.load(cache_path / model_name)
                    return True, model
                else:
                    return False, None
        else:
            return False, None


    def _write_lego_cache(self, block, mod, fold) -> None:
        cache_path = Path("lego_cache/", mkdir=True)
        model_name = f"lego_{self.source}_{self.config.data[self.source].dataset}_{mod}_{fold}.pth"
        if self.config.model_params.lego.overwrite_cache:
            torch.save(block, cache_path / model_name)
            logger.info(f"Saved LegoBlock {model_name}")
        return None


    def run(self):

        models, train_c_indeces, val_c_indeces, test_c_indeces, t_epochs = [], [], [], [], []

        for fold in range(1, self.config.folds + 1):
            logger.info(f"*****Fold {fold}*****")
            # fix seeds for reproducibility
            torch.manual_seed(fold)
            np.random.seed(fold)
            self.seed = fold

            train_data, val_data, test_data = self.get_data()
            model = self.get_model(train_data)
            wandb.watch(model)

            epoch, model, test_data, train_c_index, trainer, val_c_index = self.train_fold(fold, t_epochs, model, train_data, val_data, test_data)
            # evaluate training fold
            test_c_index, _ = trainer.evaluate(epoch, test_data, subset="test")

            # keep track of models and indices for each fold
            models.append(model)
            train_c_indeces.append(train_c_index)
            val_c_indeces.append(val_c_index)
            test_c_indeces.append(test_c_index)

            wandb.log({"mean_train_c_index": np.mean(train_c_indeces),
                       "mean_val_c_index": np.mean(val_c_indeces),
                       "std_train_c_index": np.std(train_c_indeces),
                       "std_val_c_index": np.std(val_c_indeces),
                       "mean_test_c_index": np.mean(test_c_indeces),
                       "std_test_c_index": np.std(test_c_indeces),
                       "mean_train_epoch (s)": np.mean(t_epochs)})

            if self.config.write_model:
                if self.config.write_model:
                    if self.config.model == "legoblock":
                        write_path = Path(f"block_log/{self.dataset}/{wandb.run.name}/model_checkpoint_fold_{fold}.pth")
                    else:
                        write_path = Path(f"model_log/{wandb.run.name}/model_checkpoint_fold_{fold}.pth")
                    write_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model, write_path)



        # get best model across folds
        best_fold = np.argmax(test_c_indeces)
        best_model = models[best_fold]
        # save config
        if self.config.write_model:
            torch.save({"config": self.config, "best_fold": best_fold}, write_path.parent.joinpath("metadata.pth"))
            logger.info(f"Saved models and metadata (config, best fold) to {write_path.parent}")


        wandb.finish()

        return best_model, np.mean(test_c_indeces)

    def train_fold(self, fold, t_epochs, model, train_data, val_data, test_data):
        if self.config.source in ["tcga"]:
            trainer = SurvivalTrainer(
                self.config, model, train_data, val_data, test_data, self.class_weights, fold
            )
        elif self.config.source in ["mimic", "isic"]:
            trainer = ClassificationTrainer(self.config, model, train_data, val_data, test_data, self.class_weights,
                                            fold)
        for epoch in range(1, self.config.epochs + 1):
            t_epoch, (train_c_index, val_c_index) = trainer.train_epoch(epoch)
            t_epochs.append(t_epoch)
            logger.info(f"Epoch time elapsed: {t_epoch:.3f} s")
            if trainer.stop:
                break
        return epoch, model, test_data, train_c_index, trainer, val_c_index

    def get_data(self) -> Tuple:

        data_config = self.config.data[self.source]

        data_map = {
            "tcga": TCGASurvivalDataset,
            "mimic": MimicDataset,
            "isic": ISICDataset
        }

        if self.config.model in ["healnet-early"]:
            # early fusion, so needs to flatten/concat results
            data_config["concat"] = True

        self.data = data_map[self.source](**data_config.to_dict())


        total_size = len(self.data)
        train_size = int(0.7 * total_size)
        test_size = int(0.15*total_size)
        val_size = total_size - train_size - test_size

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        train, test, val = torch.utils.data.random_split(self.data, [train_size, test_size, val_size], generator=generator)

        if self.config.model_params.class_weights == "None":
            self.class_weights = None
        else:
            self.class_weights = torch.Tensor(self._calc_class_weights(train)).to(self.device)


        # check if multiprocessing is part of config
        if self.config.sweep:
            # num_workers = 8
            num_workers = int(multiprocessing.cpu_count())
        else:
            num_workers = int(multiprocessing.cpu_count())

        # num_workers = int(multiprocessing.cpu_count())

        train_data = DataLoader(train,
                                num_workers=num_workers,
                                **self.config.loader.to_dict()
                                )

        # use different batch size for validation/test for stability
        val_args, test_args = self.config.loader.to_dict(), self.config.loader.to_dict()
        val_args["batch_size"] = len(val)  # valuate on entire val set
        test_args["batch_size"] = len(test)

        # valuate on entire val set
        logger.info(f"Using validation batch size: {val_args['batch_size']}")
        val_data = DataLoader(val,
                                num_workers=num_workers,
                                **val_args,
                                )
        test_data = DataLoader(test,
                                num_workers=num_workers,
                                **test_args,
                                )
        self._get_data_stats(self.data, train, test, val)
        return train_data, val_data, test_data



    def _get_data_stats(self, data, train, test, val):

        logger.info(f"Train samples: {int(len(train))}, Val samples: {int(len(val))}, "
                    f"Test samples: {int(len(test))}")

        if self.config.source == "tcga":
            target_distribution = lambda idx, data: dict(
                np.round(data.omic_df.iloc[idx]["y_disc"].value_counts().sort_values() / len(idx), 2))
        elif self.config.source in ["mimic", "isic"]:
            target_distribution = lambda idx, data: dict(
                np.round(data.y.iloc[idx].value_counts().sort_values() / len(idx), 2))
            # target_distribution = data.targets.bincount()

        logger.info(f"Train distribution: {target_distribution(train.indices, data)}")
        logger.info(f"Val distribution: {target_distribution(val.indices, data)}")
        logger.info(f"Test distribution: {target_distribution(test.indices, data)}")



if __name__ == '__main__':
    config = Config("config/config_dev.yml").read()
    try:
        mp_context = config.loader.multiprocessing_context  # multiprocessing context, also called in DataLoaders
        torch.multiprocessing.set_start_method(mp_context)
    except RuntimeError:
        # get multiprocessing context
        logger.info(f"Multiprocessing context already set")

    pipeline = Fusion(config=config)
    pipeline.run()


