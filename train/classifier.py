import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.classifier.model import Classifier
from model.dataloader import MRIRAMLoader, FastMRIDataset, MRIDataset


# Custom collate function for GPU optimization
def gpu_collate_fn(batch):
    """Custom collate function for efficient GPU transfer"""
    images, labels = zip(*batch)

    # Stack tensors
    images = torch.stack(images, dim=0)
    labels = torch.stack(labels, dim=0)

    return images, labels


@hydra.main(config_path='../config/model', config_name='classifier', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    datapath = cfg.dataset
    lr = cfg.lr

    logger = TensorBoardLogger(save_dir="./log", name="classifier")
    # train_ram_loader = MRIRAMLoader(datapath, 'train')
    # train_dataset = FastMRIDataset(*train_ram_loader.get_data())
    train_dataset = MRIDataset(data_path=datapath, split='train', apply_augmentation=True)
    class_weights = train_dataset.get_class_weights()
    print(f"Class weights: {class_weights}")
    weight_list = [class_weights[i] for i in sorted(class_weights.keys())]
    weight_tensor = torch.tensor(weight_list, dtype=torch.float32)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True,
        pin_memory=True,  # Faster GPU transfer
        collate_fn=gpu_collate_fn
    )
    # val_ram_loader = MRIRAMLoader(datapath, 'val')
    val_loader = DataLoader(
        # dataset=FastMRIDataset(*val_ram_loader.get_data()),
        dataset=MRIDataset(data_path=datapath, split='val'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True
    )
    model = Classifier(lr=lr, class_weights=weight_tensor, epochs=cfg.max_epoch)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
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
        devices=[0, 1],
        # overfit_batches=3,
        val_check_interval=1.0,
        logger=logger,
        # gradient_clip_val=1.0,
        precision='16-mixed',
        callbacks=[early_stop_callback],
        # accumulate_grad_batches=2,
        log_every_n_steps=5,
        enable_checkpointing=False,
    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    test_ram_loader = MRIRAMLoader(datapath, 'test')
    test_loader = DataLoader(
        dataset=FastMRIDataset(*test_ram_loader.get_data()),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
    )
    trainer.test(model=model, dataloaders=test_loader)
