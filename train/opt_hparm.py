import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from model.cvit.model import SegmentTransformer
from model.dataloader import MRIDataset


def objective(trial):
    norm = trial.suggest_categorical('normalization', ['batch', 'instance', 'layer'])
    embedding_size = trial.suggest_int('embedding_size', 64, 512, step=4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9)
    batch_size = 128
    num_workers = 3

    model = SegmentTransformer(embedding_size=embedding_size, dropout=dropout_rate, norm=norm, lr=1e-3)
    model.classification_loss = nn.CrossEntropyLoss()

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        logger=TensorBoardLogger(save_dir="./log", name="cvit_optuna"),
        val_check_interval=1.0,
        precision='16-mixed',
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        enable_checkpointing=False
    )

    train_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v4.hdf5', 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v4.hdf5', 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_accuracy = trainer.callback_metrics['val_accuracy'].max()
    return val_accuracy
