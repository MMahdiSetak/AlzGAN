import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.ddpm.diffusion import Diffusion
from model.ddpm.trainer import GaussianDiffusion
from model.ddpm.unet import create_model
from model.dataloader import DDPMPairDataset
from train.callbacks import ImageLogger, VideoLogger, MetricsLogger


@hydra.main(config_path='../config/model', config_name='ddpm', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    input_size = cfg.input_size
    depth_size = cfg.depth_size
    num_channels = cfg.num_channels
    num_res_blocks = cfg.num_res_blocks
    heads = cfg.num_heads
    save_and_sample_every = cfg.save_and_sample_every
    lr = cfg.lr
    timesteps = cfg.timesteps
    epochs = cfg.max_epoch
    in_channels = cfg.in_channels
    out_channels = cfg.out_channels

    logger = TensorBoardLogger(save_dir="./log", name="ddpm")
    train_loader = DataLoader(
        dataset=DDPMPairDataset(datapath, 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        dataset=DDPMPairDataset(datapath, 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    model = create_model(input_size, num_channels, num_res_blocks, num_heads=heads, use_checkpoint=False,
                         in_channels=in_channels, out_channels=out_channels)
    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=timesteps,  # number of steps
        loss_type='l1',  # L1 or L2
        channels=out_channels
    )
    lit_model = Diffusion(
        diffusion_model=diffusion,
        train_lr=2e-6,
        ema_decay=0.995,
        step_start_ema=2000,
        update_ema_every=10
    )
    callbacks = []
    # callbacks.append(ImageLogger(
    #     batch_frequency=750, max_images=4, clamp=True))
    # callbacks.append(VideoLogger(
    #     batch_frequency=1500, max_videos=4, clamp=True))
    callbacks.append(MetricsLogger(batch_frequency=500))
    callbacks.append(
        ModelCheckpoint(
            # dirpath="my_checkpoints/",  # custom folder
            filename="ddpm_best_model",  # custom filename pattern
            monitor="train_loss",  # metric to monitor
            mode="min",  # "min" for loss, "max" for accuracy, etc.
            save_top_k=1,  # save top 3 models instead of just 1
            # save_last=True,  # also save last epoch checkpoint
            verbose=True,
        )
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=[0],
        # val_check_interval=5000,
        check_val_every_n_epoch=10,
        # overfit_batches=20,
        limit_val_batches=10,
        logger=logger,
        # gradient_clip_val=0,
        precision='16-mixed',
        # callbacks=[EMACallback()],
        callbacks=callbacks,
        accumulate_grad_batches=2,
        log_every_n_steps=5,
        # enable_checkpointing=True,
    )
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
