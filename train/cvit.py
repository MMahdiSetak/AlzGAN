from model.cvit.model import SegmentTransformer
from model.dataloader import CustomDataset

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from torch.utils.data import DataLoader


def run():
    batch_size = 32
    num_workers = 4
    model = SegmentTransformer(batch_size)
    logger = TensorBoardLogger(save_dir="./log", name="cvit")

    # data_loader = DataLoader('dataset/mri_label_v3.hdf5', batch_size)
    # train_data_generator = data_loader.data_generator(batch_size, "train", pet=False, label=True)
    # val_data_generator = data_loader.data_generator(batch_size, "val", pet=False, label=True)

    train_loader = DataLoader(
        CustomDataset('dataset/mri_label_v3.hdf5', 'train', pet=False, label=True),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        CustomDataset('dataset/mri_label_v3.hdf5', 'train', pet=False, label=True),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True
    )
    # Set up the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="auto",
        logger=logger,
        val_check_interval=1.0,
    )

    # Train the model using the data generators
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
