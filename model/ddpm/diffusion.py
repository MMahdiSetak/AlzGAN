import copy

import torch
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio

from model.ddpm.trainer import GaussianDiffusion


class Diffusion(pl.LightningModule):
    def __init__(
            self,
            diffusion_model: GaussianDiffusion,
            train_lr: float = 2e-6,
            ema_decay: float = 0.995,
            step_start_ema: int = 2000,
            update_ema_every: int = 10,
    ):
        super().__init__()
        self.model = diffusion_model
        self.lr = train_lr
        # self.save_hyperparameters()

        # EMA model
        # self.ema = EMA(self.hparams.ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self.metrics = MetricCollection({
            "MAE": MeanAbsoluteError(),
            "MSE": MeanSquaredError(),
            "PSNR": PeakSignalNoiseRatio(data_range=1),
            # "SSIM": StructuralSimilarityIndexMeasure(),
        })

    def forward(self, x, condition_tensors=None):
        return self.model(x, condition_tensors=condition_tensors)

    def training_step(self, batch, batch_idx):
        mri, pet, _ = batch
        loss = self.model(pet, condition_tensors=mri)
        loss = loss.mean()
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mri, real_pet, _ = batch
        bs = real_pet.size(0)
        fake_pet = self.model.sample(bs, mri)

        # PSNR expects images scaled in [0, 1], so scale img and x_gt accordingly:
        # Assuming your images are in [-1, 1], rescale to [0, 1]
        fake_pet = rescale(fake_pet)
        real_pet = rescale(real_pet)

        metrics = self.metrics(fake_pet, real_pet)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return metrics

    def configure_optimizers(self):
        opt = Adam(self.model.parameters(), lr=self.lr)
        return opt


def rescale(im):
    return (im.clamp(-1, 1) + 1) / 2


class EMACallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step < pl_module.hparams.step_start_ema:
            pl_module.ema_model.load_state_dict(pl_module.model.state_dict())
        elif step % pl_module.hparams.update_ema_every == 0:
            pl_module.ema.update_model_average(pl_module.ema_model, pl_module.model)
