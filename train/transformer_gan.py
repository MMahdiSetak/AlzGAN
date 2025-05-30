from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy

from model.dataloader import DDPMPairDataset

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from model.transformer_gan.model import TransformerGAN


@hydra.main(config_path='../config/model', config_name='transformer_gan', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    model = TransformerGAN()
    logger = TensorBoardLogger(save_dir="./log", name="transformer_gan")

    train_loader = DataLoader(
        dataset=DDPMPairDataset(cfg.dataset, 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True
    )
    val_loader = DataLoader(
        dataset=DDPMPairDataset(cfg.dataset, 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_accuracy",
    #     mode="max",
    #     save_top_k=1,
    #     filename="cvit_best_model",
    # )
    # early_stop_callback = EarlyStopping(
    #     monitor='val_accuracy',
    #     patience=cfg.model.early_stop,
    #     verbose=True,
    #     mode='max'
    # )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epoch,
        accelerator="auto",
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        val_check_interval=1.0,
        precision='16-mixed',
        log_every_n_steps=5,
        enable_checkpointing=False,
        # callbacks=[early_stop_callback],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_loader = DataLoader(
        dataset=DDPMPairDataset(cfg.dataset, 'test'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    trainer.test(model=model, dataloaders=test_loader)
