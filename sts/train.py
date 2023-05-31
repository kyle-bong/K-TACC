import argparse
from tqdm.auto import tqdm
from omegaconf import OmegaConf

# pl modules
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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

    wandb_logger = WandbLogger(project='korean_linguistic_augmentation')
    
    # checkpoint config
    checkpoint_callback = ModelCheckpoint(dirpath="./saved/", 
                                          filename=f'{cfg.model.saved_name}',
                                          save_top_k=1,
                                          monitor='val_loss',
                                          mode='max')

    # Train & Test
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.max_epoch,
        log_every_n_steps=cfg.train.logging_step,
        callbacks=[checkpoint_callback],
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