import os
import sys

import transformers
from transformers import get_cosine_schedule_with_warmup
import torch
import torchmetrics
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = cfg.model.model_name
        self.lr = cfg.train.learning_rate
        self.drop_out = cfg.train.drop_out
        self.warmup_ratio = cfg.train.warmup_ratio
        self.weight_decay = cfg.train.weight_decay
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=1,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out
        )
        self.epoch = cfg.train.max_epoch
        self.batch_size = cfg.train.batch_size

        self.loss_func = torch.nn.L1Loss()
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.plm(x)['logits']

        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss, prog_bar=True, logger=True)
        pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("val_pearson", pearson_corr, prog_bar=True, logger=True)

        return {'val_loss':loss, 'val_pearson_corr':pearson_corr}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        test_pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("test_pearson", test_pearson_corr)
        return test_pearson_corr
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=self.lr*0.01, last_epoch=-1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_ratio*self.trainer.estimated_stepping_batches,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)
        # scheduler = ExponentialLR(optimizer, gamma=0.5)
        scheduler = {'scheduler':scheduler, 'interval':'step', 'frequency':1}

        return [optimizer], [scheduler]