import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.MRI2PET.model import MRI2PET
from model.dataloader import MRI2PETDataset


@hydra.main(config_path='../config/model', config_name='mri2pet', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    epochs = cfg.max_epoch
    lr = cfg.lr

    logger = TensorBoardLogger(save_dir="./log", name="mri2pet")
    train_loader = DataLoader(
        dataset=MRI2PETDataset(datapath, 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        dataset=MRI2PETDataset(datapath, 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    lit_model = MRI2PET(lr=lr)
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val",
        mode="min",
        save_top_k=1,
        filename="mri2pet_best_model",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=[0],
        # val_check_interval=5000,
        # check_val_every_n_epoch=10,
        # overfit_batches=100,
        limit_val_batches=5,
        logger=logger,
        # gradient_clip_val=0,
        precision='16-mixed',
        callbacks=[checkpoint_callback],
        accumulate_grad_batches=2,
        log_every_n_steps=5,
        enable_checkpointing=True,
    )
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
