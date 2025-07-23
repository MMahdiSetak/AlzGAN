import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, Specificity
import monai.transforms as mt
import random

from model.classifier.module import Simple3DCNN, Parametric3DCNN
from model.vq_gan_3d.vqgan import VQGAN


class CustomOneOf(mt.MapTransform):
    """
    Custom MONAI transform to mimic tio.OneOf: Selects and applies one transform based on probabilities.
    """

    def __init__(self, keys, transforms_dict):
        super().__init__(keys)
        self.transforms = list(transforms_dict.keys())
        probs = list(transforms_dict.values())
        self.probs = [p / sum(probs) for p in probs]  # Normalize to sum=1

    def __call__(self, data):
        selected_transform = random.choices(self.transforms, weights=self.probs, k=1)[0]
        return selected_transform(data)


class Classifier(pl.LightningModule):
    # def __init__(self, cfg, class_weights):
    def __init__(self, class_weights, num_layers=4, base_channels=32, channel_multiplier=2,
                 cnn_dropout_rate=0.3, fc_dropout_rate=0.6, fc_hidden=128, lr=1e-3, weight_decay=1e-2, max_epoch=300):
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

        # vq_gan_model = VQGAN.load_from_checkpoint(checkpoint_path=cfg.vq_gan_checkpoint)
        # self.encoder = vq_gan_model.encoder
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        # self.classifier = Simple3DCNN(input_size=(80, 96, 80), channels=[1, 32, 64, 128, 256], fc=128, num_classes=3,
        #                               dropout_rate=0.4)
        # self.classifier = Simple3DCNN(input_size=(8, 8, 8), channels=[64, 16], fc=64, num_classes=3,
        #                               dropout_rate=0.6)

        self.classifier = Parametric3DCNN(
            input_size=(80, 96, 80),
            num_layers=num_layers,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            cnn_dropout_rate=cnn_dropout_rate,
            fc_dropout_rate=fc_dropout_rate,
            fc_hidden=fc_hidden,
            num_classes=num_classes
        )

        # MONAI GPU-accelerated transforms (applied in steps after batch on GPU)
        keys = ['mri']  # Key for image in dict
        self.train_transform = mt.Compose([
            mt.NormalizeIntensityd(keys=keys, nonzero=True),  # Z-score with mask x > 0
            mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),  # LR flip (adjust axis if orientation differs)
            CustomOneOf(keys=keys, transforms_dict={
                mt.Rand3DElasticd(
                    keys=keys,
                    sigma_range=(5, 8),  # Controls smoothness (approx to num_control_points=7)
                    magnitude_range=(5, 10),  # Approx to max_displacement=7.5; tune based on voxel size
                    prob=1.0,
                    mode='trilinear',
                    padding_mode='zeros',
                    # device=torch.device("cuda")
                ): 0.35,
                mt.RandAdjustContrastd(
                    keys=keys,
                    gamma=(0.74, 1.35),  # Matches log_gamma=0.3 (exp(-0.3) to exp(0.3))
                    prob=1.0
                ): 0.30,
                mt.RandAffined(
                    keys=keys,
                    scale_range=(0.8, 1.2),
                    rotate_range=(0, 0, 0.2618),  # 15 degrees in radians (adjust axes as needed)
                    translate_range=8,
                    prob=1.0,
                    mode='trilinear',
                    padding_mode='zeros',
                    # device=torch.device("cuda")
                ): 0.25,
                mt.Identityd(keys=keys): 0.1
            }),
            mt.Resized(keys=keys, spatial_size=(80, 96, 80), mode='trilinear')
        ])

        self.val_test_transform = mt.Compose([
            mt.NormalizeIntensityd(keys=keys, nonzero=True),
            mt.Resized(keys=keys, spatial_size=(80, 96, 80), mode='trilinear')
        ])

    def _apply_transform_per_sample(self, batch, transform):
        # Unstack batch to list of per-sample dicts (each 'mri' is (1, 1, D, H, W))
        samples = [{'mri': mri, 'label': label} for mri, label in zip(batch['mri'], batch['label'])]
        # Apply transform to each sample independently (GPU, per-sample random)
        aug_samples = [transform(sample) for sample in samples]
        # Restack
        mri_aug = torch.cat([s['mri'].unsqueeze(0) for s in aug_samples], dim=0)
        labels_aug = torch.stack([s['label'] for s in aug_samples])  # Labels unchanged
        return mri_aug, labels_aug

    def forward(self, mri):
        # mri = self.encoder(mri)
        out = self.classifier(mri)
        return out

    def training_step(self, batch, batch_idx):
        # mri, labels = self._apply_transform_per_sample(batch, self.train_transform)

        # mri, labels = self._apply_transform_per_sample(batch, self.train_transform)
        # batch = self.train_transform(batch)
        # mri = batch['mri']
        # labels = batch['label']

        mri, labels = batch
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
        # inputs, labels = self._apply_transform_per_sample(batch, self.val_test_transform)
        # batch = self.val_test_transform(batch)
        # inputs = batch['mri']
        # labels = batch['label']
        inputs, labels = batch
        bs = len(labels)
        outputs = self(inputs)
        loss = self.classification_loss(outputs, labels)
        metrics = self.val_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # inputs, labels = self._apply_transform_per_sample(batch, self.val_test_transform)
        # batch = self.val_test_transform(batch)
        # inputs = batch['mri']
        # labels = batch['label']
        inputs, labels = batch
        bs = len(labels)
        outputs = self(inputs)
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
            T_max=self.epochs,  # Total epochs
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
