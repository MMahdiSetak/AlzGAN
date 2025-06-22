from torch.optim import AdamW
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio

from model.MRI2PET.module import DiffusionModel


class MRI2PET(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.model = DiffusionModel(num_timesteps=1000, beta_start=0.0015, beta_end=0.02, pet_image_dim=64,
                                    n_pet_channels=64, mri_image_dim=64, n_mri_channels=64, embed_dim=256,
                                    laplace=0.25)
        metrics = {
            "MAE": MeanAbsoluteError(),
            "MSE": MeanSquaredError(),
            "PSNR": PeakSignalNoiseRatio(data_range=1),
        }
        self.train_metrics = MetricCollection(metrics, postfix="/train")
        self.val_metrics = MetricCollection(metrics, postfix="/val")

    def forward(self, x, condition_tensors=None):
        self.model(condition_tensors, x, gen_loss=True, noise_level=1, include_laplace=True)

    def training_step(self, batch, batch_idx):
        mri, pet, _ = batch
        loss, _ = self.model(mri, pet, gen_loss=True, noise_level=1, include_laplace=True)
        loss = loss.mean()
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, prog_bar=False)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mri, pet, _ = batch
        loss, _ = self.model(mri, pet, gen_loss=True, noise_level=1, include_laplace=True)
        loss = loss.mean()
        self.log('loss/val', loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5,min_lr=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'loss/train_epoch',
                'interval': 'epoch',
                'frequency': 1,
            }
        }


def rescale(im):
    return (im.clamp(-1, 1) + 1) / 2
