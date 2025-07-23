import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback
import torch
from torch.utils.data import DataLoader
import gc

from model.classifier.model import Classifier
from model.dataloader import MRIDataset, MRIRAMLoader

batch_size = 32
num_workers = 24
datapath = 'dataset/mri_label_v5.1_Rigid.hdf5'
train_ram_loader = MRIRAMLoader(datapath, 'train')
train_dataset = MRIDataset(*train_ram_loader.get_data(), split='train', apply_augmentation=True)
class_weights = train_dataset.get_class_weights()
print(f"Class weights: {class_weights}")
weight_list = [class_weights[i] for i in sorted(class_weights.keys())]
weight_tensor = torch.tensor(weight_list, dtype=torch.float32)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True,
)
val_ram_loader = MRIRAMLoader(datapath, 'val')
val_loader = DataLoader(
    dataset=MRIDataset(*val_ram_loader.get_data(), split='val'),
    batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False, persistent_workers=True,
)


class MaxAccuracyCallback(Callback):
    def __init__(self):
        super().__init__()
        self.max_val_accuracy = float('-inf')

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get current val_accuracy from logged metrics
        current_val_accuracy = trainer.callback_metrics.get('accuracy/val', torch.tensor(0.0)).item()
        self.max_val_accuracy = max(self.max_val_accuracy, current_val_accuracy)


def objective(trial):
    num_layers = trial.suggest_int('num_layers', 3, 5)  # 3-5 layers to avoid too deep/overfit
    base_channels = trial.suggest_categorical('base_channels', [8, 16, 32, 64])  # Starting channels
    channel_multiplier = trial.suggest_categorical('channel_multiplier', [1, 1.2, 1.5, 2])  # Growth factor
    cnn_dropout_rate = trial.suggest_float('cnn_dropout_rate', 0., 0.5, step=0.1)  # Dropout for regularization
    fc_dropout_rate = trial.suggest_float('fc_dropout_rate', 0.2, 0.7, step=0.1)  # Dropout for regularization
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)  # Log scale for LR
    fc_hidden = trial.suggest_categorical('fc_hidden', [64, 128, 256])  # FC size, small options

    model = Classifier(
        class_weights=weight_tensor, num_layers=num_layers, base_channels=base_channels,
        channel_multiplier=channel_multiplier, cnn_dropout_rate=cnn_dropout_rate, fc_dropout_rate=fc_dropout_rate,
        fc_hidden=fc_hidden, lr=lr, weight_decay=1e-2, max_epoch=300
    )
    max_acc_callback = MaxAccuracyCallback()
    trainer = pl.Trainer(
        max_epochs=5,
        num_sanity_val_steps=0,
        accelerator="auto",
        val_check_interval=1.0,
        logger=TensorBoardLogger(save_dir="./log", name="classifier_optuna"),
        gradient_clip_val=1.0,
        precision='16-mixed',
        log_every_n_steps=5,
        enable_checkpointing=False,
        callbacks=[max_acc_callback]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    val_accuracy = max_acc_callback.max_val_accuracy
    if val_accuracy == float('-inf'):  # Fallback if not logged
        val_accuracy = trainer.callback_metrics.get('accuracy/val', 0.0).item()
    del trainer, model
    torch.cuda.empty_cache()
    gc.collect()
    return val_accuracy


def run():
    study = optuna.create_study(study_name='classification', direction="maximize", storage="sqlite:///class.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=100, gc_after_trial=True)

    best_params = study.best_params
    print(f"Best Hyperparameters: {best_params}")
