"""
Training loop utils for different discriminative tasks
"""
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
# from torcheval.metrics import AUC
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from mm_lego.utils import Config, setup_logging, timed_function
from mm_lego.models import EarlyStopping
from mm_lego.models.losses import nll_loss, l1_loss
from sksurv.metrics import concordance_index_censored
import wandb
from transformers import get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup

logger = setup_logging()
# use this wandb to be able to toggle off

class BaseTrainer:

    def __init__(self,
                 config: Config,
                 model: nn.Module,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 test_data: DataLoader,
                 class_weights: np.ndarray,
                 fold: int):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.class_weights = class_weights
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.fold = fold

        # Initial setup
        self.optimizer = optim.Adam(model.parameters(), **self.config.optimizer.to_dict())

        if self.config.scheduler.type == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=int(self.config.early_stopping.patience/2), factor=0.5)
        else:
            self.scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_training_steps = len(self.train_data) * self.config.schedule.epochs,
                                                             num_warmup_steps = 1,)


        self.early_stopping = EarlyStopping(**self.config.early_stopping.to_dict())
        # binary variable denoting whether to stop loop
        self.stop = False
        self.model.train()

    def train_epoch(self):
        pass

    def eval_epoch(self):
        pass


class ClassificationTrainer(BaseTrainer):
    def __init__(self,
                 config: Config,
                 model: nn.Module,
                 train_data: DataLoader,
                 val_data: DataLoader,
                 test_data: DataLoader,
                 class_weights: np.ndarray,
                 fold: int):
        super().__init__(config, model, train_data, val_data, test_data, class_weights, fold)
        # self.multi_class = multi_class
        # check num classes
        self.num_classes = len(np.unique(train_data.dataset.dataset.targets))
        self.multi_class = self.num_classes > 2



    @timed_function(unit="s")
    def train_epoch(self, epoch: int):
        train_loss = 0.0
        preds = []
        labels = []
        all_logits = []

        for batch, (features, y) in enumerate(tqdm(self.train_data)):
            features = [feat.to(self.device) for feat in features]
            y = y.to(self.device)

            if batch == 0 and epoch == 0:
                logger.info(f"Modality shapes: ")
                [logger.info(feat.shape) for feat in features]
                logger.info(f"Modality dtypes:")
                [logger.info(feat.dtype) for feat in features]

            self.optimizer.zero_grad()

            if self.config.model == "multimodn":
                num_classes = y.bincount().shape[0]
                _, logits = self.model.forward(features, F.one_hot(y, num_classes=num_classes))
            else:
                logits = self.model(features)


            y_hat = torch.argmax(logits, dim=1)
            all_logits.extend(logits.detach().cpu().numpy())
            preds.extend(y_hat.cpu().numpy())
            labels.extend(y.cpu().numpy())
            # calc cross-entropy

            loss = F.cross_entropy(logits, y, weight=self.class_weights)

            reg_loss = l1_loss(self.model, **self.config.reg.to_dict())

            loss_value = loss.item()

            train_loss += loss_value + reg_loss

            loss += reg_loss
            current_lr = self.optimizer.param_groups[0]['lr']

            # backprop
            loss.backward()
            self.optimizer.step()

            wandb.log({
                f"fold_{self.fold}_train_loss": train_loss/(batch+1),
                f"fold_{self.fold}_lr": current_lr,
            })

            step = wandb.run.step
            val_metric, val_loss = self.evaluate(epoch, self.val_data, subset="val", step=step)
            if self.config.scheduler.type == "ReduceLROnPlateau":
                self.scheduler.step(metrics=val_loss)
            else:
                self.scheduler.step()
            # evaluate

            if self.stop:
                break

        # calc auc
        if self.multi_class:
            one_hot_labels = F.one_hot(torch.tensor(labels, device="cpu"), num_classes=self.num_classes)
            train_metric = roc_auc_score(one_hot_labels, all_logits, multi_class="ovr", average="macro")
        else:
            train_metric = roc_auc_score(labels, preds)

        train_loss /= len(self.train_data)

        logger.info(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_c_index: {train_metric:.4f}')
        wandb.log({f"fold_{self.fold}_train_c_index": train_metric})

        return train_metric, val_metric


    def evaluate(self, epoch, data, subset: str = "val", step: int = None):
        self.model.eval()
        preds = []
        labels = []
        all_logits = []
        val_loss = 0.0

        for batch, (features, y) in enumerate(data):
            features = [feat.to(self.device) for feat in features]
            y = y.to(self.device)

            with torch.no_grad():
                if self.config.model == "multimodn":
                    num_classes = y.bincount().shape[0]
                    _, logits = self.model.forward(features, F.one_hot(y, num_classes=num_classes))
                else:
                    logits = self.model(features)
                all_logits.extend(logits.detach().cpu().numpy())

                y_hat = torch.argmax(logits, dim=1)
                preds.extend(y_hat.cpu().numpy().tolist())
                labels.extend(y.cpu().numpy().tolist())

                loss = F.cross_entropy(logits, y, weight=self.class_weights)
                reg_loss = l1_loss(self.model, **self.config.reg.to_dict())

                loss_value = loss.item()
                val_loss += loss_value + reg_loss

        val_loss /= len(data)
        if self.multi_class:
            one_hot_labels = F.one_hot(torch.tensor(labels, device="cpu"), num_classes=self.num_classes)
            val_metric = roc_auc_score(one_hot_labels, all_logits, multi_class="ovr", average="macro")
        else:
            val_metric = roc_auc_score(labels, preds)

        # log losses
        logger.info(f'Epoch: {epoch}, {subset}_loss: {val_loss:.4f}, {subset}_c_index: {val_metric:.4f}')
        wandb.log({f"fold_{self.fold}_{subset}_c_index": val_metric,
                     f"fold_{self.fold}_{subset}_loss": val_loss,
                     },
                    step=step)

        if subset == "val":
            if self.early_stopping.step(val_loss, self.model):
                self.stop = True
                self.model = self.early_stopping.load_best_weights(self.model)
        self.model.train()
        return val_metric, val_loss


class SurvivalTrainer(BaseTrainer):

    def __init(self,
               config: Config,
               model: nn.Module,
               train_data: DataLoader,
               val_data: DataLoader,
               test_data: DataLoader,
               class_weights: np.ndarray,
               fold: int):
        super().__init__(config, model, train_data, val_data, test_data, class_weights, fold)

    @timed_function(unit="s")
    def train_epoch(self, epoch: int):
        risk_scores = []
        censorships = []
        event_times = []
        train_loss_surv, train_loss = 0.0, 0.0
        n_batches = len(self.train_data)

        for batch, (features, censorship, event_time, y_disc) in enumerate(tqdm(self.train_data)):
            features = [feat.to(self.device) for feat in features] # list of tensors
            censorship = censorship.to(self.device)
            event_time = event_time.to(self.device)
            y_disc = y_disc.to(self.device)

            if batch == 0 and epoch == 0:
                logger.info(f"Modality shapes: ")
                [logger.info(feat.shape) for feat in features]
                logger.info(f"Modality dtypes:")
                [logger.info(feat.dtype) for feat in features]

            # zero gradients for every batch
            self.optimizer.zero_grad()

            if self.config.model == "multimodn":
            # note that we need to pass the target here (just for multimodn) for intermediate loss
                model_loss, logits = self.model.forward(features, F.one_hot(y_disc, num_classes=4))
            else:
                logits = self.model(features)
                model_loss = 0.0
            y_hat = torch.topk(logits, k=1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1-hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()  # risk = -sum(survival)

            # using Negative Log Likelihood survival loss
            loss = nll_loss(hazards=hazards, S=survival, Y=y_disc, c=censorship, weights=self.class_weights)

            # regularisation loss term
            reg_loss = l1_loss(self.model, **self.config.reg.to_dict())

            risk_scores.append(risk)
            censorships.append(censorship.detach().cpu().numpy())
            event_times.append(event_time.detach().cpu().numpy())

            loss_value = loss.item()

            # cumulative epoch loss (for logging)
            train_loss_surv += loss_value
            train_loss += loss_value + reg_loss

            # what's being optimised
            loss += reg_loss

            current_lr = self.optimizer.param_groups[0]['lr']

            # backprop
            loss.backward()
            # step optimizer and scheduler
            self.optimizer.step()

            wandb.log({
                f"fold_{self.fold}_train_loss": train_loss/(batch+1),
                f"fold_{self.fold}_train_surv_loss": train_loss_surv/(batch+1),
                f"fold_{self.fold}_lr": current_lr,
            })
            step = wandb.run.step # use this step for other logging calls

            # evaluate
            # if batch % self.config.eval_frequency == 0:
            val_c_index, val_loss = self.evaluate(epoch, self.val_data, subset="val", step=step)
            if self.config.scheduler.type == "ReduceLROnPlateau":
                self.scheduler.step(metrics=val_loss)
            else:
                self.scheduler.step()



            if self.stop:
                break
                # test_c_index = self.evaluate(epoch, self.test_data, subset="test", step=step)

        risk_scores_full = np.concatenate(risk_scores)
        censorships_full = np.concatenate(censorships)
        event_times_full = np.concatenate(event_times)
        # calculate c-Index for epoch
        train_c_index = concordance_index_censored((1-censorships_full).astype(bool), event_times_full, risk_scores_full, tied_tol=1e-08)[0]


        # take mean loss for logging
        train_loss /= len(self.train_data)
        train_loss_surv /= len(self.train_data)


        # log for debugging (note that steps are logged earlier for wandb)
        logger.info(f'Epoch: {epoch}, train_loss: {train_loss:.4f}, train_surv_loss, {train_loss_surv:.4f}, train_c_index: {train_c_index:.4f}')

        wandb.log({f"fold_{self.fold}_train_c_index": train_c_index,
                   },  step=step)


        # return for logging and early stopping
        return train_c_index, val_c_index


    def evaluate(self, epoch: int, data: DataLoader, subset: str = "val", step: int = None):

        self.model.eval()

        risk_scores = []
        censorships = []
        event_times = []
        predictions = []
        labels = []
        val_loss_surv, val_loss = 0.0, 0.0
        train_batches = len(self.train_data)

        for batch, (features, censorship, event_time, y_disc) in enumerate(data):
            # only move to GPU now (use CPU for preprocessing)
            # if missing_mode is not None:  # handle for missing modality ablation
            #     features, use_omic = self._sample_missing(features, use_omic, missing_mode)
            features = [feat.to(self.device) for feat in features]
            censorship = censorship.to(self.device)
            event_time = event_time.to(self.device)
            y_disc = y_disc.to(self.device)

            # don't store gradients for validation
            with torch.no_grad():
                if self.config.model == "multimodn":
                # note that we need to pass the target here (just for multimodn) for intermediate loss
                    _, logits = self.model.forward(features, F.one_hot(y_disc, num_classes=4))
                else:
                    logits = self.model(features)
                    # if subset == "test":
                    #     print(logits)

                hazards = torch.sigmoid(logits)
                survival = torch.cumprod(1 - hazards, dim=1)
                risk = -torch.sum(survival, dim=1).detach().cpu().numpy()

                loss = nll_loss(hazards=hazards, S=survival, Y=y_disc, c=censorship, weights=self.class_weights)

                reg_loss = l1_loss(self.model, **self.config.reg.to_dict())

                # log risk, censorship and event time for concordance index
                risk_scores.append(risk)
                censorships.append(censorship.detach().cpu().numpy())
                event_times.append(event_time.detach().cpu().numpy())

                loss_value = loss.item()
                val_loss_surv += loss_value
                val_loss += loss_value + reg_loss

                predictions.append(logits.argmax(1).cpu().tolist())
                labels.append(y_disc.detach().cpu().tolist())

        # predictions = np.concatenate(predictions)
        # labels = np.concatenate(labels)

        val_loss_surv /= len(data)
        val_loss /= len(data)

        risk_scores_full = np.concatenate(risk_scores)
        censorships_full = np.concatenate(censorships)
        event_times_full = np.concatenate(event_times)

        # calculate epoch-level concordance index
        val_c_index = \
        concordance_index_censored((1 - censorships_full).astype(bool), event_times_full, risk_scores_full)[0]

        # Log losses
        logger.info(f'Epoch: {epoch}, {subset}_loss: {val_loss:.4f}, {subset}_surv_loss: {val_loss_surv:.4f}, {subset}_c_index: {val_c_index:.4f}')
        wandb.log({f"fold_{self.fold}_{subset}_c_index": val_c_index,
                   f"fold_{self.fold}_{subset}_loss": val_loss,
                   f"fold_{self.fold}_{subset}_surv_loss": val_loss_surv,
                   },
                  step=step)

        # if self.config.early_stopping.on and subset == "val" and :
        if subset == "val":
            if self.early_stopping.step(val_loss, self.model):
                self.stop = True
                self.model = self.early_stopping.load_best_weights(self.model)

        self.model.train()
        return val_c_index, val_loss


