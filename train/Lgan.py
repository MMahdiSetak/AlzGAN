import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.Lgan.model import LGAN
from model.dataloader import LcDDPMDataset
from train.callbacks import ImageLogger, VideoLogger, MetricsLogger


@hydra.main(config_path='../config/model', config_name='LGAN', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    epochs = cfg.max_epochs

    logger = TensorBoardLogger(save_dir="./log", name="Lgan")
    train_loader = DataLoader(
        dataset=LcDDPMDataset(datapath, 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        dataset=LcDDPMDataset(datapath, 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    # model = create_model(input_size, num_channels, num_res_blocks, "", num_heads=heads, use_checkpoint=False,
    #                      in_channels=in_channels, out_channels=out_channels)
    # diffusion = GaussianDiffusion(
    #     model,
    #     image_size=input_size,
    #     depth_size=depth_size,
    #     timesteps=timesteps,  # number of steps
    #     loss_type='l1',  # L1 or L2
    #     channels=out_channels
    # )
    model = LGAN(cfg)
    callbacks = []
    callbacks.append(ModelCheckpoint(
        monitor='train/loss',
        save_top_k=1, mode='min', filename='LGAN_best_model')
    )
    # # callbacks.append(ImageLogger(
    # #     batch_frequency=750, max_images=4, clamp=True))
    # # callbacks.append(VideoLogger(
    # #     batch_frequency=1500, max_videos=4, clamp=True))
    # callbacks.append(MetricsLogger(batch_frequency=50))
    # callbacks.append(
    #     ModelCheckpoint(
    #         # dirpath="my_checkpoints/",  # custom folder
    #         filename="ddpm_best_model",  # custom filename pattern
    #         monitor="train_loss",  # metric to monitor
    #         mode="min",  # "min" for loss, "max" for accuracy, etc.
    #         save_top_k=1,  # save top 3 models instead of just 1
    #         # save_last=True,  # also save last epoch checkpoint
    #         verbose=True,
    #     )
    # )
    trainer = pl.Trainer(
        max_epochs=epochs,
        num_sanity_val_steps=0,
        accelerator="auto",
        # strategy="fsdp",
        # devices=[0, 1, 2],
        # val_check_interval=5000,
        check_val_every_n_epoch=10,
        # overfit_batches=20,
        limit_val_batches=5,
        logger=logger,
        # gradient_clip_val=0,
        precision='16-mixed',
        gradient_clip_val=1,
        # callbacks=[EMACallback()],
        callbacks=callbacks,
        # accumulate_grad_batches=2,
        log_every_n_steps=5,
        # enable_checkpointing=True,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_loader = DataLoader(
        dataset=LcDDPMDataset(datapath, 'test'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    model_path = os.path.join(logger.log_dir, "checkpoints", "LGAN_best_model.ckpt")
    model = LGAN.load_from_checkpoint(model_path)
    trainer.test(model, test_loader)
