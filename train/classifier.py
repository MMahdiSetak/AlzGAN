import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.classifier.model import Classifier
from model.dataloader import FastMRIDataset, MRIRAMLoader


@hydra.main(config_path='../config/model', config_name='classifier', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    lr = cfg.lr

    logger = TensorBoardLogger(save_dir="./log", name="classifier")
    ram_loader = MRIRAMLoader(datapath, 'train')
    train_loader = DataLoader(
        dataset=FastMRIDataset(*ram_loader.get_data(), transform=True),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True
    )
    val_loader = DataLoader(
        dataset=FastMRIDataset(datapath, 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True
    )
    model = Classifier(lr=lr)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        filename="classifier_best_model",
    )
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=cfg.early_stop,
        verbose=True,
        mode='max'
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epoch,
        # strategy=DDPStrategy(find_unused_parameters=True),
        num_sanity_val_steps=0,
        accelerator="auto",
        devices=[3, 4],
        val_check_interval=1.0,
        logger=logger,
        gradient_clip_val=1.0,
        precision='16-mixed',
        callbacks=[early_stop_callback],
        # accumulate_grad_batches=2,
        log_every_n_steps=5,
        enable_checkpointing=False,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_loader = DataLoader(
        dataset=FastMRIDataset(datapath, 'test'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    trainer.test(model=model, dataloaders=test_loader)
