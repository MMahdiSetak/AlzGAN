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
        dataset=MRIDataset('dataset/mri_label_v3.hdf5', 'train'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        dataset=MRIDataset('dataset/mri_label_v3.hdf5', 'val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="auto",
        logger=logger,
        val_check_interval=1.0,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
