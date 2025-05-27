import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, Specificity
import torch.nn.functional as F
import monai.transforms as T

from model.gan_class.module import Generator, Classifier


class GANClass(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(GANClass, self).__init__()
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

        self.generator = Generator()
        self.classifier = Classifier()
        self.train_transforms = T.Compose([
            T.RandRotate(range_x=np.pi / 18, range_y=np.pi / 18, range_z=np.pi / 18, prob=0.5),
            T.Rand3DElastic(sigma_range=(2, 5), magnitude_range=(0.1, 0.3), prob=0.3),
            T.RandAffine(translate_range=(10, 10, 10), scale_range=(-0.1, 0.1), prob=0.5),
            T.RandGaussianNoise(std=0.01, prob=0.2),
            T.RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),
            T.RandBiasField(prob=0.3)
        ])

    def forward(self, image):
        enc_features, bottleneck_features = self.generator(image)
        stacked_features = torch.cat((enc_features, bottleneck_features), dim=1)
        return self.classifier(stacked_features)

    def training_step(self, batch, batch_idx):
        mri, labels = batch  # Batch is (N, D, H, W), already on GPU
        mri = mri.unsqueeze(1)  # (N, 1, D, H, W)
        mri = mri / 255.0  # Scale to [0, 1]

        # Apply transforms to each sample individually
        transformed_mri = []
        for i in range(mri.shape[0]):
            sample = mri[i]  # Extract single sample: (1, D, H, W)
            transformed_sample = self.train_transforms(sample)  # Apply transforms
            transformed_mri.append(transformed_sample)

        # Stack transformed samples back into a batch
        mri = torch.stack(transformed_mri, dim=0)  # (N, 1, D, H, W)

        mri = mri * 2 - 1  # Scale to [-1, 1]
        mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        bs = len(labels)
        outputs = self(mri)
        loss = self.classification_loss(outputs, labels)

        lr = self.optimizers().param_groups[0]['lr']
        metrics = self.train_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        mri, labels = batch
        mri = mri.unsqueeze(1)  # (N, 1, D, H, W)
        mri = mri / 127.5 - 1  # Scale to [-1, 1]
        mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        bs = len(labels)
        outputs = self(mri)
        metrics = self.val_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        mri, labels = batch
        mri = mri.unsqueeze(1)  # (N, 1, D, H, W)
        mri = mri / 127.5 - 1  # Scale to [-1, 1]
        mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        bs = len(labels)
        outputs = self(mri)
        metrics = self.test_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

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
