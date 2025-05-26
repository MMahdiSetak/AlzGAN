from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, Specificity
from monai.networks.nets import SwinUNETR

from model.classifier.module import DenseNet3DClassifier


class Classifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(Classifier, self).__init__()
        self.lr = lr
        self.classification_loss = nn.CrossEntropyLoss()
        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=3),
            "precision": Precision(task="multiclass", num_classes=3),
            "recall": Recall(task="multiclass", num_classes=3),
            "f1_score": F1Score(task="multiclass", num_classes=3),
            "auc": AUROC(task="multiclass", num_classes=3),
            "specificity": Specificity(task="multiclass", num_classes=3),
        }
        self.train_metrics = MetricCollection(metrics, prefix="train_")
        self.val_metrics = MetricCollection(metrics, prefix="val_")
        self.test_metrics = MetricCollection(metrics, prefix="test_")
        self.model = DenseNet3DClassifier(num_classes=3)

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        mri, labels = batch
        bs = len(labels)
        outputs = self(mri)
        loss = self.classification_loss(outputs, labels)

        lr = self.optimizers().param_groups[0]['lr']
        metrics = self.train_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        bs = len(labels)
        outputs = self(inputs)
        metrics = self.val_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        bs = len(labels)
        outputs = self(inputs)
        metrics = self.test_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        }
