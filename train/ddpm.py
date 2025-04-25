import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.ddpm.diffusion import Diffusion
from model.ddpm.trainer import GaussianDiffusion
from model.ddpm.unet import create_model
from model.dataloader import DDPMPairDataset


@hydra.main(config_path='../config/model', config_name='ddpm', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    input_size = cfg.input_size
    depth_size = cfg.depth_size
    num_channels = cfg.num_channels
    num_res_blocks = cfg.num_res_blocks
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
    model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels,
                         use_fp16=True)
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
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        logger=logger,
        # gradient_clip_val=0,
        precision='32',
        # callbacks=[EMACallback()],
        accumulate_grad_batches=2,
        log_every_n_steps=5,
        enable_checkpointing=False,
    )
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
