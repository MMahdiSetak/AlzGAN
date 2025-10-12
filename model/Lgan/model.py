import torch
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from model.ddpm.unet import create_model
from model.vq_vae_3d.vqvae import VQVAE


class LGAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = create_model(
            cfg.input_size, cfg.num_channels, cfg.num_res_blocks, cfg.channel_mult, num_heads=cfg.num_heads,
            attention_resolutions=cfg.attention_resolutions,
            use_checkpoint=False, in_channels=cfg.in_channels, out_channels=cfg.out_channels
        )

        mri_vqgan_model = VQVAE.load_from_checkpoint(checkpoint_path=cfg.mri_vqgan_checkpoint)
        pet_vqgan_model = VQVAE.load_from_checkpoint(checkpoint_path=cfg.pet_vqgan_checkpoint)
        self.mri_encoder = mri_vqgan_model.encoder
        self.mri_encoder.eval()
        # self.pet_encoder = pet_vqgan_model.encoder
        # self.pet_encoder.eval()
        self.pet_decoder = pet_vqgan_model.decoder
        self.pet_decoder.eval()

        for param in self.mri_encoder.parameters():
            param.requires_grad = False
        # for param in self.pet_encoder.parameters():
        #     param.requires_grad = False
        for param in self.pet_decoder.parameters():
            param.requires_grad = False

        self.save_hyperparameters()

        metrics = {
            "MAE": MeanAbsoluteError(),
            "MSE": MeanSquaredError(),
            "PSNR": PeakSignalNoiseRatio(data_range=1),
            "SSIM": StructuralSimilarityIndexMeasure(data_range=1)
        }
        self.train_metrics = MetricCollection(metrics, prefix="train/")
        self.val_metrics = MetricCollection(metrics, prefix="val/")
        self.test_metrics = MetricCollection(metrics, prefix="test/")

    def forward(self, x):
        x = self.mri_encoder(x)
        zero_step = torch.zeros(x.shape[0]).to(self.device)
        x = self.model(x, zero_step)
        x = self.pet_decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        mri, real_pet = batch
        fake_pet = self(mri)
        loss = F.mse_loss(real_pet, fake_pet)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mri, real_pet = batch
        bs = real_pet.size(0)
        fake_pet = self(mri).detach()
        loss = F.mse_loss(real_pet.detach(), fake_pet)

        metrics = self.val_metrics(fake_pet, real_pet)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return metrics

    def test_step(self, batch, batch_idx):
        mri, real_pet = batch
        bs = real_pet.size(0)
        fake_pet = self(mri).detach()
        loss = F.mse_loss(real_pet.detach(), fake_pet)

        metrics = self.test_metrics(fake_pet, real_pet)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return metrics

    @torch.no_grad()
    def inference(self, mri):
        return self(mri).detach()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            # fused=True
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
