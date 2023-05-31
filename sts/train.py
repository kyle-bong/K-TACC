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
                              name=cfg.model.saved_name,
                              log_model="all")
    # checkpoint config
    checkpoint_callback = ModelCheckpoint(dirpath='saved/', 
                                          monitor='val_loss',
                                          mode='min',
                                          filename=f'{cfg.model.saved_name}',
                                          save_top_k=2)

    # Train & Test
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.max_epoch,
        log_every_n_steps=1,
        val_check_interval=0.5,
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