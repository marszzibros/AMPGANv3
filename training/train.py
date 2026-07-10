import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

from models import AMPGANv3, GANCheckpoint
from data_utils import AMPDatasets, AMPDataModule

import time
import sys
import os

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))

    generator_cfg = cfg.model.generator
    discriminator_cfg = [cfg.GAN, cfg.MIC]

    generator_opt_cfg = cfg.optimizer.generator
    discriminator_opt_cfg = [cfg.disc1, cfg.disc2]

    dataset = AMPDatasets(data_path="data/", max_length=68)

    datamodule = AMPDataModule(data_path=cfg.data_path, max_length=cfg.max_length, batch_size=cfg.batch_size)
    model = AMPGANv3(latent_dim=cfg.latent_dim, 
                     amp_dataset=dataset, 
                     generator_cfg=generator_cfg,
                     discriminator_cfg=discriminator_cfg,
                     generator_opt_cfg=generator_opt_cfg,
                     discriminator_opt_cfg=discriminator_opt_cfg,
                     run_num=cfg.run)

    # Initialize the logger
    wandb_logger = WandbLogger(
        project="AMPGANv3",
        save_dir="logs/",
        name=f"AMPGANv3_{cfg.run}",
        log_model="None",
        offline=False
    )
    trainer = L.Trainer(**cfg.trainer, logger=wandb_logger, callbacks=[GANCheckpoint(every_n_epochs=50, run_num=cfg.run), LearningRateMonitor(logging_interval='epoch')])
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()