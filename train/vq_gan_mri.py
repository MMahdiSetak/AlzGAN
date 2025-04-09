"""Adapted from https://github.com/FirasGit/medicaldiffusion"""

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from model.log import Logger
from model.vq_gan_3d.vqgan import VQGAN
from model.dataloader import DataLoader
from train.callbacks import ImageLogger, VideoLogger
import hydra
from omegaconf import DictConfig, open_dict


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu / 8.) * (bs / 4.) * base_lr
    print(
        "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
            cfg.model.lr, accumulate, ngpu / 8, bs / 4, base_lr))

    model = VQGAN(cfg)
    # model.to("cuda")
    # logger = Logger("vq_gan_3d")
    # logger.save_model_metadata(model, logger.pet_target_shape, "vq_gan_3d", 2)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss',
                                     save_top_k=3, mode='min', filename='latest_checkpoint'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                                     save_top_k=-1, filename='{epoch}-{step}-{train/recon_loss:.2f}'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=-1,
                                     filename='{epoch}-{step}-10000-{train/recon_loss:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=750, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=1500, max_videos=4, clamp=True))

    # load the most recent checkpoint file
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                cfg.model.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      cfg.model.resume_from_checkpoint)

    trainer = pl.Trainer(
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        accelerator='auto',
    )

    data_loader = DataLoader('dataset/mri_pet_label_v3.hdf5', bs)
    train_dataloader = data_loader.mri_generator(bs, "train")
    val_dataloader = data_loader.mri_generator(bs, "val")
    trainer.fit(model, train_dataloader, val_dataloader)
