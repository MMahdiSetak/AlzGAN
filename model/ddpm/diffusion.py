from torch.optim import AdamW
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        metrics = {
            "MAE": MeanAbsoluteError(),
            "MSE": MeanSquaredError(),
            "PSNR": PeakSignalNoiseRatio(data_range=1),
            # "SSIM": StructuralSimilarityIndexMeasure(),
        }
        self.train_metrics = MetricCollection(metrics, postfix="/train")
        self.val_metrics = MetricCollection(metrics, postfix="/val")

    def forward(self, x, condition_tensors=None):
        return self.model(x, condition_tensors=condition_tensors)

    def training_step(self, batch, batch_idx):
        mri, pet, _ = batch
        # pet, mri, _ = batch
        loss = self.model(pet, condition_tensors=mri)
        loss = loss.mean()
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mri, real_pet, _ = batch
        # real_pet, mri, _ = batch
        bs = real_pet.size(0)
        fake_pet = self.model.sample(bs, mri)

        # PSNR expects images scaled in [0, 1], so scale img and x_gt accordingly:
        # Assuming your images are in [-1, 1], rescale to [0, 1]
        fake_pet = rescale(fake_pet)
        real_pet = rescale(real_pet)

        metrics = self.val_metrics(fake_pet, real_pet)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return metrics

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


def rescale(im):
    return (im.clamp(-1, 1) + 1) / 2


class EMACallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step < pl_module.hparams.step_start_ema:
            pl_module.ema_model.load_state_dict(pl_module.model.state_dict())
        elif step % pl_module.hparams.update_ema_every == 0:
            pl_module.ema.update_model_average(pl_module.ema_model, pl_module.model)
