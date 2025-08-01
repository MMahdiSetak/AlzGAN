import os.path

import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.classifier.model import Classifier
from model.dataloader import MRIDataset, MRIRAMLoader, MergedDataset


@hydra.main(config_path='../config/model', config_name='classifier', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    mri_dataset = cfg.mri_dataset

    logger = TensorBoardLogger(save_dir="./log", name="classifier")
    # train_ram_loader = MRIRAMLoader(mri_dataset, 'train')
    # train_dataset = FastMRIDataset(*train_ram_loader.get_data())
    # train_dataset = MRIDataset(data_path=datapath, split='train')
    # train_dataset = MRIDataset(*train_ram_loader.get_data(), split='train', apply_augmentation=True)
    train_dataset = MergedDataset(csv_path=cfg.tabular_dataset, hdf5_path=mri_dataset, split='train',
                                  mri_cache=None, apply_augmentation=True)
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    weight_list = [class_weights[i] for i in sorted(class_weights.keys())]
    weight_tensor = torch.tensor(weight_list, dtype=torch.float32)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
        persistent_workers=True,
        # pin_memory=True,
    )
    val_ram_loader = MRIRAMLoader(mri_dataset, 'val')
    val_loader = DataLoader(
        # dataset=FastMRIDataset(*val_ram_loader.get_data()),
        # dataset=MRIDataset(*val_ram_loader.get_data(), split='val'),
        dataset=MergedDataset(csv_path=cfg.tabular_dataset, hdf5_path=mri_dataset, split='val',
                              mri_cache=val_ram_loader.get_data()),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True,
    )
    model = Classifier(
        num_layers=cfg.num_layers,
        base_channels=cfg.base_channels,
        channel_multiplier=cfg.channel_multiplier,
        cnn_dropout_rate=cfg.cnn_dropout_rate,
        fc_dropout_rate=cfg.fc_dropout_rate,
        fc_hidden=cfg.fc_hidden,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_epoch=cfg.max_epoch,
        class_weights=weight_tensor,
        vq_gan_checkpoint=cfg.vq_gan_checkpoint
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="accuracy/val",
        mode="max",
        save_top_k=1,
        filename="classifier_best_model",
    )
    early_stop_callback = EarlyStopping(
        monitor='accuracy/val',
        patience=cfg.early_stop,
        verbose=True,
        mode='max'
    )
    trainer = pl.Trainer(
        max_epochs=cfg.max_epoch,
        # strategy=DDPStrategy(find_unused_parameters=True),
        num_sanity_val_steps=0,
        accelerator="auto",
        # strategy="fsdp",
        # TODO make it config
        # devices=[0, 1],
        # overfit_batches=3,
        val_check_interval=1.0,
        logger=logger,
        gradient_clip_val=1.0,
        precision='16-mixed',
        callbacks=[early_stop_callback, checkpoint_callback],
        # accumulate_grad_batches=2,
        log_every_n_steps=5,
        enable_checkpointing=True,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # test_ram_loader = MRIRAMLoader(datapath, 'test')
    test_loader = DataLoader(
        # dataset=MRIDataset(data_path=datapath, split='test'),
        # dataset=MRIDataset(*test_ram_loader.get_data(), split='test'),
        dataset=MergedDataset(csv_path=cfg.tabular_dataset, hdf5_path=mri_dataset, split='test'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )
    os.path.join(logger.log_dir, "checkpoints", "classifier_best_model.ckpt")
    model = Classifier.load_from_checkpoint(logger.log_dir)
    trainer.test(model=model, dataloaders=test_loader)
