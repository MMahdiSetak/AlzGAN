"""Adapted from https://github.com/FirasGit/medicaldiffusion"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from model.dataloader import VQGANDataset
from model.vq_gan_3d.vqgan import VQGAN
from train.callbacks import ImageLogger, VideoLogger
import hydra
from omegaconf import DictConfig, open_dict


@hydra.main(config_path='../config/model', config_name='vq_gan_3d', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = batch_size, cfg.lr, cfg.gpus, cfg.accumulate_grad_batches
    with open_dict(cfg):
        cfg.lr = accumulate * (ngpu / 8.) * (bs / 4.) * base_lr
    print(
        "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
            cfg.lr, accumulate, ngpu / 8, bs / 4, base_lr))

    model = VQGAN(cfg)

    callbacks = []
    # callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
    #                                  save_top_k=3, mode='min', filename='latest_checkpoint'))
    # callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
    #                                  save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    # callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
    #                                  filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=1500, max_videos=4, clamp=True))

    # load the most recent checkpoint file
    # base_dir = os.path.join(cfg.default_root_dir, 'lightning_logs')
    # if os.path.exists(base_dir):
    #     log_folder = ckpt_file = ''
    #     version_id_used = 0
    #     for folder in os.listdir(base_dir):
    #         version_id = int(folder.split('_')[1])
    #         if version_id > version_id_used:
    #             version_id_used = version_id
    #             log_folder = folder
    #     if len(log_folder) > 0:
    #         ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
    #         for fn in os.listdir(ckpt_folder):
    #             if fn == 'latest_checkpoint.ckpt':
    #                 ckpt_file = 'latest_checkpoint_prev.ckpt'
    #                 os.rename(os.path.join(ckpt_folder, fn),
    #                           os.path.join(ckpt_folder, ckpt_file))
    #         if len(ckpt_file) > 0:
    #             cfg.resume_from_checkpoint = os.path.join(
    #                 ckpt_folder, ckpt_file)
    #             print('will start from the recent ckpt %s' %
    #                   cfg.resume_from_checkpoint)

    logger = TensorBoardLogger(save_dir="./log", name="vq_gan_mri")
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        # overfit_batches=5,
        # default_root_dir=cfg.default_root_dir,
        logger=logger,
        callbacks=callbacks,
        # max_steps=cfg.max_steps,
        max_epochs=cfg.max_epochs,
        precision=cfg.precision,
        accelerator='auto',
        # devices=[0],
        # strategy=DDPStrategy(find_unused_parameters=True)
    )

    # data_loader = DataLoader('dataset/mri_pet_label_v3.hdf5', bs)
    # train_dataloader = data_loader.mri_generator(bs, "train")
    # val_dataloader = data_loader.mri_generator(bs, "val")
    train_loader = DataLoader(
        dataset=VQGANDataset(datapath, 'train', modality='mri'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        dataset=VQGANDataset(datapath, 'val', modality='mri'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    trainer.fit(model, train_loader, val_loader)
