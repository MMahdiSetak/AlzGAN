import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from model.dataloader import MRIDataset
from model.diff_class.model import DiffClass


@hydra.main(config_path='../config/model', config_name='diff_class', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    lr = cfg.lr

    logger = TensorBoardLogger(save_dir="./log", name="diff_class")
    train_loader = DataLoader(
        dataset=MRIDataset(datapath, 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        dataset=MRIDataset(datapath, 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    model = DiffClass(embedding_size=128, dropout=0.2, lr=1e-3)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        filename="cvit_best_model",
    )
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=cfg.model.early_stop,
        verbose=True,
        mode='max'
    )
    trainer = pl.Trainer(
        max_epochs=cfg.model.max_epoch,
        # num_sanity_val_steps=0,
        accelerator="auto",
        val_check_interval=1,
        logger=logger,
        gradient_clip_val=1.0,
        precision='16-mixed',
        callbacks=[early_stop_callback],
        accumulate_grad_batches=2,
        log_every_n_steps=5,
        enable_checkpointing=False,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_loader = DataLoader(
        dataset=MRIDataset(cfg.model.dataset, 'test'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    trainer.test(model=model, dataloaders=test_loader)
