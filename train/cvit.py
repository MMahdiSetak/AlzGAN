from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.cvit.model import SegmentTransformer
from model.dataloader import MRIDataset

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.model.batch_size
    num_workers = cfg.model.num_workers
    model = SegmentTransformer(embedding_size=cfg.model.embedding_size, dropout=cfg.model.dropout_rate, lr=cfg.model.lr,
                               norm=cfg.model.norm)
    logger = TensorBoardLogger(save_dir="./log", name="cvit")

    train_loader = DataLoader(
        dataset=MRIDataset(cfg.model.dataset, 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        dataset=MRIDataset(cfg.model.dataset, 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )

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
        accelerator="auto",
        logger=logger,
        val_check_interval=1.0,
        precision='16-mixed',
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        callbacks=[early_stop_callback],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v4.hdf5', 'test'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    trainer.test(model=model, dataloaders=test_loader)
