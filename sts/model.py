import os
import sys

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

class Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = cfg.model.model_name
        self.lr = cfg.train.learning_rate
        self.drop_out = cfg.train.drop_out
        self.warmup_ratio = cfg.train.warmup_ratio
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=1,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out
        )

        self.loss_func = torch.nn.L1Loss()
        self.optimizer = torch.optim.AdamW(params=self.parameters(), lr=self.lr)

    def forward(self, x):
        x = self.plm(x)['logits']

        return x
    
    def training_step(self, batch, batch_idx):
        print('batch: ', batch)
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze().float())
        self.log("val_pearson", pearson_corr)

        return {'val_loss':loss, 'val_pearson_corr':pearson_corr}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        test_pearson_corr = torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze())
        self.log("test_pearson", test_pearson_corr)
        return test_pearson_corr
    
    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = transformers.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_ratio * self.trainer.estimated_stepping_batches,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        scheduler = {'scheduler':scheduler, 'interval':'step', 'frequency':1}

        return [optimizer], [scheduler]