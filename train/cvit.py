from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.cvit.model import SegmentTransformer
from model.dataloader import MRIDataset

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader


def run():
    batch_size = 32
    num_workers = 4
    model = SegmentTransformer(batch_size)
    logger = TensorBoardLogger(save_dir="./log", name="cvit")

    train_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v4.hdf5', 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v4.hdf5', 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        filename="cvit_best_model",
        verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        verbose=True,
        mode='max'
    )
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="auto",
        logger=logger,
        val_check_interval=1.0,
        callbacks=[checkpoint_callback, early_stop_callback],
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    test_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v4.hdf5', 'test'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )
    trainer.test(model=model, dataloaders=test_loader)
