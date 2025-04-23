import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pytest
from mm_lego.utils import Config
from mmhb.loader import TCGASurvivalDataset



# @pytest.fixture(scope="module")
# def args():
#
#     config = Config("config/config_dev.yml").read()
#
#     # Create a dataset
#     dummy_dataset = TCGASurvivalDataset(data_path=config.data_path,
#                                         dataset="kirp")
#     # Split the dataset
#     train_data, val_data, test_data = random_split(dummy_dataset, [70, 15, 15])
#
#     # Create dataloaders
#     train_data = DataLoader(train_data, batch_size=32, shuffle=True)
#     val_data = DataLoader(val_data, batch_size=32, shuffle=False)
#     test_data = DataLoader(test_data, batch_size=32, shuffle=False)
#
#
#
#
#     return config, train_data, val_data, test_data