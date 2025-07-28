import torch
from torch.optim import AdamW
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model.ddpm.trainer import GaussianDiffusion
from model.ddpm.unet import create_model
from model.vq_gan_3d.vqgan import VQGAN


class LcDDPM(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        model = create_model(
            cfg.input_size, cfg.num_channels, cfg.num_res_blocks, "1,2,4", num_heads=cfg.num_heads,
            use_checkpoint=False, in_channels=cfg.in_channels, out_channels=cfg.out_channels
        )
        self.model = GaussianDiffusion(
            model,
            image_size=cfg.input_size,
            depth_size=cfg.depth_size,
            timesteps=cfg.timesteps,  # number of steps
            loss_type='l1',  # L1 or L2
            channels=cfg.out_channels
        )

        mri_vqgan_model = VQGAN.load_from_checkpoint(checkpoint_path=cfg.mri_vqgan_checkpoint)
        pet_vqgan_model = VQGAN.load_from_checkpoint(checkpoint_path=cfg.pet_vqgan_checkpoint)
        self.mri_encoder = mri_vqgan_model.encoder
        self.mri_encoder.eval()
        self.pet_encoder = pet_vqgan_model.encoder
        self.pet_encoder.eval()
        self.pet_decoder = pet_vqgan_model.decoder
        self.pet_decoder.eval()

        for param in self.mri_encoder.parameters():
            param.requires_grad = False
        for param in self.pet_encoder.parameters():
            param.requires_grad = False
        for param in self.pet_decoder.parameters():
            param.requires_grad = False

        self.save_hyperparameters()

        metrics = {
            "MAE": MeanAbsoluteError(),
            "MSE": MeanSquaredError(),
            "PSNR": PeakSignalNoiseRatio(data_range=2),
            "SSIM": StructuralSimilarityIndexMeasure(data_range=2)
        }
        self.train_metrics = MetricCollection(metrics, prefix="train/")
        self.val_metrics = MetricCollection(metrics, prefix="val/")

    def forward(self, x, condition_tensors=None):
        condition_tensors = self.mri_encoder(condition_tensors)
        x = self.pet_encoder(x)
        return self.model(x, condition_tensors=condition_tensors)

    def training_step(self, batch, batch_idx):
        mri, pet = batch
        loss = self(pet, condition_tensors=mri)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mri, real_pet = batch
        bs = real_pet.size(0)
        fake_pet = self.inference(mri)

        metrics = self.val_metrics(fake_pet, real_pet)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        return metrics

    @torch.no_grad()
    def inference(self, mri):
        bs = mri.size(0)
        enc_mri = self.mri_encoder(mri)
        enc_pet = self.model.sample(bs, enc_mri)
        fake_pet = self.pet_decoder(enc_pet)
        return fake_pet

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            fused=True
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.max_epochs,  # Total epochs
            eta_min=self.cfg.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }
