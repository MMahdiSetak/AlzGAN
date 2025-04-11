from model.cvit.model import SegmentTransformer
from model.dataloader import DataLoader

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


def run():
    model = SegmentTransformer()
    logger = TensorBoardLogger(save_dir="./log", name="cvit")

    batch_size = 2
    data_loader = DataLoader('dataset/mri_pet_label_v3.hdf5', batch_size)
    train_data_generator = data_loader.data_generator(batch_size, "train", pet=False, label=True)
    val_data_generator = data_loader.data_generator(batch_size, "val", pet=False, label=True)

    # Set up the PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator="cpu",
        logger=logger,
        val_check_interval=1.0,
    )

    # Train the model using the data generators
    trainer.fit(model=model, train_dataloaders=train_data_generator, val_dataloaders=val_data_generator)
