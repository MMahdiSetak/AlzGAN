"""Adapted from https://github.com/FirasGit/medicaldiffusion"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from model.dataloader import VQGANDataset
from model.vq_gan_3d.vqgan import VQGAN
from train.callbacks import ImageLogger, VideoLogger
import hydra
from omegaconf import DictConfig


@hydra.main(config_path='../config/model', config_name='vq_gan_3d', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    image_type = cfg.image_type

    model = VQGAN(cfg)
    callbacks = []
    callbacks.append(ModelCheckpoint(
        monitor='val/recon_loss',
        save_top_k=1, mode='min', filename='latest_checkpoint')
    )
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=1500, max_videos=4, clamp=True))

    logger = TensorBoardLogger(save_dir="./log", name=f"vq_gan_{image_type}")
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        check_val_every_n_epoch=10,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        # overfit_batches=5,
        logger=logger,
        log_every_n_steps=5,
        callbacks=callbacks,
        # max_steps=cfg.max_steps,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        accelerator='auto',
        # devices=[0],
        sync_batchnorm=True,
        strategy=DDPStrategy(find_unused_parameters=False)
    )

    train_loader = DataLoader(
        dataset=VQGANDataset(datapath, 'train', modality=image_type),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True
    )
    val_loader = DataLoader(
        dataset=VQGANDataset(datapath, 'val', modality=image_type),
        batch_size=batch_size, num_workers=num_workers // 8 if num_workers > 8 else 2, shuffle=False, drop_last=False,
        persistent_workers=True
    )
    trainer.fit(model, train_loader, val_loader)
