import argparse
from tqdm.auto import tqdm
from omegaconf import OmegaConf

# pl modules
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

# Dataloader & Model
from dataloader import Dataloader
from model import Model
import wandb
from pytorch_lightning.loggers import WandbLogger

def main(cfg):
    # Load dataloader & model
    dataloader = Dataloader(cfg)
    model = Model(cfg)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    wandb_logger = WandbLogger(project='korean_linguistic_augmentation',
                              name=f'{cfg.model.saved_name}_{cfg.train.learning_rate}',
                              log_model="all")
    # checkpoint config
    checkpoint_callback = ModelCheckpoint(dirpath='saved/', 
                                          monitor='val_pearson',
                                          mode='max',
                                          filename=f'{cfg.model.saved_name}',
                                          save_top_k=0)

    # Train & Test
    trainer = pl.Trainer(accelerator="gpu", max_epochs=cfg.train.max_epoch,
        log_every_n_steps=cfg.train.logging_step,
        val_check_interval=1.0,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=wandb_logger)
    
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)
    wandb.finish()
    
if __name__ == '__main__':
    # receive arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(f'./{args.config}.yaml')

    # seed
    seed_everything(cfg.train.seed)

    # main
    main(cfg)