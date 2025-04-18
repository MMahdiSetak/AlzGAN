import pytorch_lightning as pl
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.gan.discriminator import Discriminator
from model.gan.generator import Generator
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio


class GAN(pl.LightningModule):
    def __init__(self, lr=1e-5):
        super().__init__()
        self.automatic_optimization = False

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.criterion = nn.BCELoss()
        self.pixelwise_loss = nn.MSELoss()

        self.lr = lr

        self.metrics = MetricCollection({
            "MAE": MeanAbsoluteError(),
            "MSE": MeanSquaredError(),
            "PSNR": PeakSignalNoiseRatio(data_range=1),
            # "SSIM": StructuralSimilarityIndexMeasure(),
        })

    def training_step(self, batch, batch_idx):
        real_mri, real_pet = batch
        bs = real_pet.shape[0]

        opt_g, opt_d = self.optimizers()

        valid = torch.ones((bs, 1), device=self.device)
        fake = torch.zeros((bs, 1), device=self.device)
        # -------------------------
        #  Train Discriminator
        # -------------------------
        fake_pet = self.generator(real_mri)
        d_loss_real = self.criterion(self.discriminator(real_pet), valid)
        d_loss_fake = self.criterion(self.discriminator(fake_pet.detach()), fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        # -------------------------
        #  Train Generator
        # -------------------------
        g_loss = self.criterion(self.discriminator(fake_pet), valid) + self.pixelwise_loss(fake_pet, real_pet)
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        return {"g_loss": g_loss, "d_loss": d_loss}

    def validation_test(self, batch, batch_idx):
        real_mri, real_pet = batch
        bs = real_pet.shape[0]
        fake_pet = self.generator(real_mri)
        metrics = self.metrics(fake_pet, real_pet)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        return metrics

    def validation_step(self, batch, batch_idx):
        return self.validation_test(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.validation_test(batch, batch_idx)

    def configure_optimizers(self):
        opt_g = optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # scheduler = {
        #     'scheduler': ReduceLROnPlateau(opt_g, mode='min', patience=3, factor=0.5),
        #     'monitor': 'train_g_loss',
        # }
        # Return both optimizers and the scheduler (applied to generator)
        return [opt_g, opt_d]  # , [scheduler]

    # def on_train_epoch_end(self):
    #     # Optionally, visualize progress by saving generated and real images.
    #     # For illustration, we use the first sample from a dummy batch created on the fly.
    #     # In practice, you might want to store a sample from training_step or use a dedicated callback.
    #     self.generator.eval()
    #     # Create a dummy input. You may want to use a fixed validation batch instead.
    #     dummy_mri = torch.randn((1, *mri_target_shape), device=self.device)
    #     fake_pet = self.generator(dummy_mri)
    #     real_mri = dummy_mri
    #     # Convert the first channel of the first sample to NumPy array for visualization
    #     fake_pet_img = fake_pet[0][0].detach().cpu().numpy()
    #     real_mri_img = real_mri[0][0].detach().cpu().numpy()
    #     visualize_progress(fake_pet_img, f"epoch_{self.current_epoch:03d}_fake_pet")
    #     visualize_progress(real_mri_img, f"epoch_{self.current_epoch:03d}_real_mri")
    #     self.generator.train()
