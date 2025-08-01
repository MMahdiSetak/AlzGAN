import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, Specificity

from model.classifier.module import Simple3DCNN, Parametric3DCNN
from model.vq_gan_3d.vqgan import VQGAN


class Classifier(pl.LightningModule):
    def __init__(self, class_weights, num_layers=4, base_channels=32, channel_multiplier=2,
                 cnn_dropout_rate=0.3, fc_dropout_rate=0.6, fc_hidden=128, lr=1e-3, weight_decay=1e-2,
                 vq_gan_checkpoint=None, max_epoch=300):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = max_epoch
        num_classes = 3

        self.classification_loss = nn.CrossEntropyLoss(weight=class_weights)

        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            "precision": Precision(task="multiclass", num_classes=num_classes),
            "recall": Recall(task="multiclass", num_classes=num_classes),
            "f1_score": F1Score(task="multiclass", num_classes=num_classes),
            "auc": AUROC(task="multiclass", num_classes=num_classes),
            "specificity": Specificity(task="multiclass", num_classes=num_classes),
        }
        self.train_metrics = MetricCollection(metrics, postfix="/train")
        self.val_metrics = MetricCollection(metrics, postfix="/val")
        self.test_metrics = MetricCollection(metrics, postfix="/test")

        if vq_gan_checkpoint is not None:
            vq_gan_model = VQGAN.load_from_checkpoint(checkpoint_path=vq_gan_checkpoint)
            self.encoder = vq_gan_model.encoder
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            self.mri_features = Simple3DCNN(input_size=(8, 8, 8), channels=[32, 16], fc=64, num_classes=32,
                                            dropout_rate=0.4)

        self.classifier = Parametric3DCNN(
            input_size=(80, 96, 80),
            num_layers=num_layers,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            cnn_dropout_rate=cnn_dropout_rate,
            fc_dropout_rate=fc_dropout_rate,
            fc_hidden=fc_hidden,
            num_classes=32
        )

        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, mri):
        latent_mri = self.encoder(mri)
        mri_features = self.mri_features(latent_mri)
        main_features = self.classifier(mri)
        out = self.fc(torch.cat([mri_features, main_features], dim=1))
        return out

    def training_step(self, batch, batch_idx):
        mri = batch['mri']
        labels = batch['label']
        tabular = batch['tabular']

        bs = len(labels)
        outputs = self(mri)
        loss = self.classification_loss(outputs, labels)

        lr = self.optimizers().param_groups[0]['lr']
        metrics = self.train_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('loss/train', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mri = batch['mri']
        labels = batch['label']
        tabular = batch['tabular']

        bs = len(labels)
        outputs = self(mri)
        loss = self.classification_loss(outputs, labels)
        metrics = self.val_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        mri = batch['mri']
        labels = batch['label']
        tabular = batch['tabular']

        bs = len(labels)
        outputs = self(mri)
        metrics = self.test_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,
            eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }
