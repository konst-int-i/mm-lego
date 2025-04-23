from __future__ import annotations
import torch
from mm_lego.utils import setup_logging

logger = setup_logging()



class EarlyStopping:
    def __init__(self, patience=5, verbose=False, mode='min'):
        """
        Constructor for early stopping.

        Parameters:
        - patience (int): How many epochs to wait before stopping once performance stops improving.
        - verbose (bool): If True, prints out a message for each validation metric improvement.
        - mode (str): One of ['min', 'max']. Minimize (e.g., loss) or maximize (e.g., accuracy) the metric.
        """
        assert mode in ['min', 'max'], "Mode must be 'min' or 'max'"
        self.patience = patience
        self.verbose = verbose
        self.counter = 0

        if mode == 'min':
            self.best_metric = float('inf')
            self.operator = torch.lt
        else:
            self.best_metric = float('-inf')
            self.operator = torch.gt

        self.best_model_weights = None
        self.should_stop = False

    def step(self, metric, model):
        """
        Check the early stopping conditions.

        Parameters:
        - metric (float): The latest validation metric (loss, accuracy, etc.).
        - model (torch.nn.Module): The model being trained.

        Returns:
        - bool: True if early stopping conditions met, False otherwise.
        """
        if type(metric) == float: # convert to tensor if necessary
            metric = torch.tensor(metric)

        if self.operator(metric, self.best_metric):
            if self.verbose:
                logger.info(f"Validation metric improved from {self.best_metric:.4f} to {metric:.4f}. Saving model weights.")
            self.best_metric = metric
            self.counter = 0
            self.best_model_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"Validation metric did not improve. Patience: {self.counter}/{self.patience}.")
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def load_best_weights(self, model):
        """
        Load the best model weights.

        Parameters:
        - model (torch.nn.Module): The model to which the best weights should be loaded.
        """
        if self.verbose:
            logger.info(f"Loading best model weights with validation metric value: {self.best_metric:.4f}")
        model.load_state_dict(self.best_model_weights)
        return model
