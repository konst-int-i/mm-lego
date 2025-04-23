import os
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
from mm_lego.utils import Config, setup_logging
from mm_lego.models import HealNet, MCAT, MILAttentionNet, SNN, MOTCAT, LateFusion, LegoBlock, LegoFuse, LegoMerge, Perceiver
from mm_lego.models.baselines.multimodn import *
import numpy as np
from box import Box
from typing import Optional
from mm_lego.utils.config import update_config
import wandb


logger = setup_logging()


class PipelineBase:

    def __init__(self,
                 config: Box,
                 run_name: Optional[str] = None,
                 run_stamp: Optional[str] = None,
                 **kwargs
                 ):
        """

        Args:
            config (Box): main config file
            source (str): broader data collection source, e.g., "tcga" or "mimic". Note that each source can have
                multiple datasets, which are expected to be specified in the `config_dev.yml`
            **kwargs:
        """
        self.config = config
        self._wandb_setup(run_name)
        self.source = self.config.source
        self._check_config()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mp_context = config.loader.multiprocessing_context

        # for uniform wandb logging (must be after wandb setup)
        self.config["dataset"] = self.config.data[self.config.source].dataset
        self.config["modalities"] = self.config.data[self.config.source].modalities
        self.dataset = self.config.dataset
        self.n_modalities = len(self.config.data[self.source].modalities)

        # update logged config (particularly relevant for sweep)
        if self.config.wandb:
            wandb.config.update(self.config, allow_val_change=True)

        logger.info(f"Running \n "
                    f"MODEL: {self.config.model.upper()} \n "
                    f"SOURCE: {self.source} \n "
                    f"DATASET: {self.dataset} \n"
                    f"MODALITIES: {self.config.data[self.source].modalities}")

    def _wandb_setup(self, run_name: str = None):

        wb_mode = "online" if self.config.wandb else "disabled"

        if self.config.sweep:
            logger.info("Running in sweep mode")
            # init if not initialised already

            wandb.init(
                project="mm-lego",
                mode=wb_mode,
                name=run_name,
                group=self.__class__.__name__,
                reinit=True, # reinit to avoid conflicts
            )

            self.config = update_config(self.config, wandb.config)

            # vice versa, update the wandb config to log all parameters in individual runs (not just sweep)
            # wandb.config.update(self.config)
        else:
            logger.info("Running single wandb run")
            wandb.init(project="mm-lego",
                      resume=False,
                      config=self.config.to_dict(),
                      name=run_name,
                      group=self.__class__.__name__,
                      mode=wb_mode,
                        reinit=True
                    )


    def _check_config(self):
        single_mod_models = ["snn", "amil", "healnet", "legoblock", "snn-block", "amil-block"]
        multi_mod_models = ["legofuse", "legomerge", "legomerge-tune", "healnet", "mcat", "porpoise", "multimodn", "healnet-early", "snn-amil-cc", "snn-amil-bl"]
        valid_models = list(set(multi_mod_models + single_mod_models))
        assert self.config.model in valid_models, f"Invalid model config, should be one of {valid_models}"

        # map correct modalities to be run for single_mod_models
        # tabular unimodal
        if self.config.model in ["snn", "snn-block"]:
            self.config.data[self.source].modalities = ["tab"]
            self.n_modalities = 1
        # img unimodal
        if self.config.model in ["amil", "amil-block"]:
            if self.config.source in ["tcga", "isic"]:
                self.config.data[self.source].modalities = ["img"]
            elif self.config.source in ["mimic"]:
                self.config.data[self.source].modalities = ["ts"]
            self.n_modalities = 1
        # if self.config.model in ["legoblock"]:
        #     assert len(self.config.data[self.source].modalities) == 1, "Only one modality allowed for legoblock"

        valid_sources = self.config.data.keys()
        assert self.source in valid_sources, f"Invalid data source for pipeline {self.__class__.__name__}, must be one of {valid_sources}"

    def main(self):
        self.get_data()
        self.run_model()
        self.run_eval()

    def get_data(self):
        pass

    def get_model(self, train_data: DataLoader) -> nn.Module:
        tensors = next(iter(train_data))[0]
        # note: each tensor is of shape [b, t_c, t_d]
        n_mods = len(self.config.data[self.source].modalities)

        if self.config.model == "legofuse":
            # unimodal lego
            if n_mods == 1:
                model = LegoBlock(
                    in_shape = tensors[0].shape[1:],
                    num_classes = len(self.data.targets.unique()),
                    name = self.config.data[self.source].modalities[0],
                    **self.config.model_params.lego.to_dict(),
                )
            else:
                model = LegoFuse(
                    blocks = self.blocks,
                    **self.config.model_params.lego.to_dict() # note: only few parameters apply
                )
        if self.config.model in ["legomerge", "legomerge-tune"]:
            if n_mods == 1:
                model = LegoBlock(
                    in_shape=tensors[0].shape[1:],
                    num_classes=len(self.data.targets.unique()),
                    name=self.config.data[self.source].modalities[0],
                    **self.config.model_params.lego.to_dict(),
                )
            else:
                model = LegoMerge(
                    blocks=self.blocks,
                    **self.config.model_params.lego.to_dict()
                )
        if self.config.model == "legoblock":
            model = LegoBlock(
                in_shape = tensors[0].shape[1:],
                num_classes = len(self.data.targets.unique()),
                name = self.config.data[self.source].modalities[0],
                **self.config.model_params.lego.to_dict(),
            )

        if self.config.model == "healnet":
            channels = [t.shape[2] for t in tensors]
            model = HealNet(
                modalities=len(tensors),
                input_channels=channels,
                input_axes=[1 for t in tensors],
                num_classes=len(self.data.targets.unique()),
                **self.config.model_params.healnet.to_dict()
            )

        elif self.config.model == "healnet-early":
            model = HealNet(
                input_channels=[t.shape[2] for t in tensors],
                input_axes=[1],
                modalities=1,
                num_classes=len(self.data.targets.unique()),
                **self.config.model_params.healnet.to_dict()
            )


        elif self.config.model == "mcat":
            n_classes = len(self.data.targets.unique())
            omic_shape = tensors[0].squeeze().shape[1:]
            wsi_shape = tensors[1].shape[1:]

            model = MCAT(
                n_classes=n_classes,
                omic_shape=omic_shape, # note: doesn't expect expand_dims
                wsi_shape=wsi_shape
            )

        elif self.config.model == "motcat":
            n_classes = len(self.data.targets.unique())
            omic_shape = tensors[0].squeeze().shape[1:]
            wsi_shape = tensors[1].shape[1:]

            model = MOTCAT(
                n_classes=n_classes,
                omic_shape=omic_shape,  # note: doesn't expect expand_dims
                wsi_shape=wsi_shape
            )

        elif self.config.model == "amil":
            model = MILAttentionNet(
                input_dim=tensors[0].shape[1:], # assuming image-only tensor and filtering batch dim
                n_classes = len(self.data.targets.unique()),
                size_arg = self.config.source,
                **self.config.model_params.amil.to_dict()
            )

        elif self.config.model == "snn":
            model = SNN(
                input_dim=tensors[0].shape[2], # t_d
                n_classes=len(self.data.targets.unique()),
                **self.config.model_params.snn.to_dict()
            )
        elif self.config.model == "snn-amil-cc":
            model = LateFusion(
                input_dims=(tensors[0].shape[2], tensors[1].shape[1:]),
                method="concat",
                n_classes=len(self.data.targets.unique())
            )
        elif self.config.model == "snn-amil-bl":
            model = LateFusion(
                input_dims=(tensors[0].shape[2], tensors[1].shape[1:]),
                method="bilinear",
                n_classes=len(self.data.targets.unique())
            )
        elif self.config.model == "snn-block":
            enc = SNN(
                input_dim=tensors[0].shape[2], # t_d
                n_classes=len(self.data.targets.unique()),
                final_head=False,
                **self.config.model_params.snn.to_dict()
            )
            model = LegoBlock(
                in_shape=tensors[0].shape[1:],
                num_classes=len(self.data.targets.unique()),
                name=self.config.data[self.source].modalities[0],
                encoder=enc,
                **self.config.model_params.lego.to_dict(),
            )
        elif self.config.model == "amil-block":
            enc = MILAttentionNet(
                input_dim=tensors[0].shape[1:], # assuming image-only tensor and filtering batch dim
                n_classes = len(self.data.targets.unique()),
                size_arg = self.config.source,
                final_head=False,
                **self.config.model_params.amil.to_dict()
            )
            model = LegoBlock(
                in_shape=tensors[0].shape[1:],
                num_classes=len(self.data.targets.unique()),
                name=self.config.data[self.source].modalities[0],
                encoder=enc,
                **self.config.model_params.lego.to_dict(),
            )

        elif self.config.model == "multimodn":
            l_d = self.config.model_params.multimodn.latent_dim
            layer_dims = self.config.model_params.multimodn.layer_dims

            if len(tensors) == 2:
                tab_features = tensors[0].shape[2]
                patch_dims = tensors[1].shape[2]

                encoders = [
                    MLPEncoder(state_size=l_d, hidden_layers=layer_dims, n_features=tab_features),
                    PatchEncoder(state_size=l_d, hidden_layers=layer_dims, n_features=patch_dims)
                ]
            else: # unimodal
                if self.config.data[self.source].modalities == ["tab"]:
                    tab_features = tensors[0].shape[2]
                    encoders = [
                        MLPEncoder(state_size=l_d, hidden_layers=layer_dims, n_features=tab_features)
                    ]
                else:
                    patch_dims = tensors[0].shape[2]
                    encoders = [PatchEncoder(state_size=l_d, hidden_layers=layer_dims, n_features=patch_dims)]

            decoders = [ClassDecoder(state_size=l_d, n_classes=len(self.data.targets.unique()), activation=torch.sigmoid)]

            model = MultiModNModule(
                state_size=l_d,
                encoders=encoders,
                decoders=decoders,
                **self.config.model_params.multimodn.to_dict()
            )

        logger.info(f"Model parameters: {get_model_parameters(model)}")
        model.float()
        model.to(self.device)
        return model

    def run_eval(self):
        pass

    def _calc_class_weights(self, train: Subset) -> np.ndarray:

        if self.config.model_params.class_weights in ["inverse", "inverse_root"]:
            train_targets = np.array(train.dataset.targets)[train.indices]
            _, counts = np.unique(train_targets, return_counts=True)
            if self.config.model_params.class_weights == "inverse":
                class_weights = 1. / counts
            elif self.config.model_params.class_weights == "inverse_root":
                class_weights = 1. / np.sqrt(counts)
        else:
            class_weights = None
        return class_weights



def get_model_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)