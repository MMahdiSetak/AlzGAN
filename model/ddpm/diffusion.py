import copy
from torch.optim import Adam
import pytorch_lightning as pl

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

    def forward(self, x, condition_tensors=None):
        return self.model(x, condition_tensors=condition_tensors)

    def training_step(self, batch, batch_idx):
        x, cond, _ = batch
        loss = self.model(x, condition_tensors=cond)
        loss = loss.mean()
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = Adam(self.model.parameters(), lr=self.lr)
        return opt


class EMACallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step < pl_module.hparams.step_start_ema:
            pl_module.ema_model.load_state_dict(pl_module.model.state_dict())
        elif step % pl_module.hparams.update_ema_every == 0:
            pl_module.ema.update_model_average(pl_module.ema_model, pl_module.model)
