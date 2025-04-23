"""
Run domain transfer experiment
"""
from mm_lego.pipeline import PipelineBase
from mm_lego.utils import setup_logging, Config
from mm_lego.models import HealNet
from mm_lego.models.train_tasks import SurvivalTrainer
from mmhb.loader import TCGASurvivalDataset
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

logger = setup_logging()

class DomainTransfer(PipelineBase):
    def __init__(self, config, **kwargs):

        super().__init__(config, **kwargs)
        # TODO - move to config
        self.train_sites = ["brca", "kirp", "ucec"]
        self.test_site = "blca"

    def run(self):

        models, train_c_indeces, val_c_indeces, test_c_indeces, t_epochs = [], [], [], [], []

        for fold in range(1, self.config.folds + 1):
            logger.info(f"*****Fold {fold}*****")

            torch.manual_seed(fold)
            np.random.seed(fold)

            train_data, val_data, test_data = self.get_data()

            pass

    def get_data(self):
        data_config = self.config.data[self.source]

        # get train datasets
        pass

    # omic overlap??


if __name__ == "__main__":
    config = Config("config/config_dev.yml").read()

    dt = DomainTransfer(config)