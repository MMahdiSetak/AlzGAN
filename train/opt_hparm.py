import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader

from model.cvit.model import SegmentTransformer
from model.dataloader import MRIDataset


def objective(trial):
    # Suggest hyperparameters
    embedding_size = trial.suggest_int('embedding_size', 64, 512, step=4)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.9)
    batch_size = 256
    num_workers = 3

    # Define the model with the suggested hyperparameters
    model = SegmentTransformer(embedding_size=embedding_size, dropout=dropout_rate, lr=1e-3)
    model.classification_loss = nn.CrossEntropyLoss()

    # Define trainer
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="auto",
        logger=TensorBoardLogger(save_dir="./log", name="cvit_optuna"),
        val_check_interval=1.0,
        precision='16-mixed',
        gradient_clip_val=1.0,
    )

    # Set up the DataLoader objects for train, val, and test sets
    train_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v3.hdf5', 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v3.hdf5', 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Return the validation accuracy as the objective to maximize
    val_accuracy = trainer.callback_metrics['val_accuracy'].item()
    print(val_accuracy)
    return val_accuracy
