import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, Specificity

from model.LcDDPM.model import LcDDPM
from model.classifier.module import Simple3DCNN, Parametric3DCNN
from model.classifier.tabular import TabularMLP
from model.vq_vae_3d.vqvae import VQVAE


class Classifier(pl.LightningModule):
    def __init__(self, class_weights, num_layers=4, base_channels=32, channel_multiplier=2,
                 cnn_dropout_rate=0.3, fc_dropout_rate=0.6, fc_hidden=128, embed_dim=32, lr=1e-3, eta_min=1e-5,
                 weight_decay=1e-2, vq_gan_checkpoint=None, tabular=True, ddpm_checkpoint=None, max_epoch=300,
                 num_classes=3):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.eta_min = eta_min
        self.weight_decay = weight_decay
        self.epochs = max_epoch

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

        self.classifier = Parametric3DCNN(
            input_size=(160, 192, 160),
            num_layers=num_layers,
            base_channels=base_channels,
            channel_multiplier=channel_multiplier,
            cnn_dropout_rate=cnn_dropout_rate,
            fc_dropout_rate=fc_dropout_rate,
            fc_hidden=fc_hidden,
            num_classes=embed_dim
        )
        fc_input = embed_dim

        if vq_gan_checkpoint is not None:
            vq_gan_model = VQVAE.load_from_checkpoint(checkpoint_path=vq_gan_checkpoint)
            self.encoder = vq_gan_model.encoder
            # for param in self.encoder.parameters():
            #     param.requires_grad = False
            self.mri_features = Simple3DCNN(input_size=(16, 16, 16), channels=[32, 16], fc=64, num_classes=embed_dim,
                                            dropout_rate=0.4)
            fc_input += embed_dim

        if tabular:
            self.tabular_features = TabularMLP(input_dim=13, hidden_dims=[128, 64], output_dim=embed_dim,
                                               dropout=fc_dropout_rate)
            fc_input += embed_dim
        if ddpm_checkpoint is not None:
            ddpm_model = LcDDPM.load_from_checkpoint(checkpoint_path=ddpm_checkpoint)
            self.ddpm_model = ddpm_model.model
            self.pet_features = Simple3DCNN(input_size=(16, 16, 16), channels=[32, 16], fc=64, num_classes=embed_dim,
                                            dropout_rate=0.4)
            fc_input += embed_dim

        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(fc_dropout_rate),
            nn.Linear(fc_input, num_classes)
        )

    def forward(self, mri, tabular=None):
        main_feats = self.classifier(mri)
        feats = [main_feats]
        if hasattr(self, "mri_features"):
            latent_mri = self.encoder(mri)
            mri_feats = self.mri_features(latent_mri)
            feats.append(mri_feats)
            if hasattr(self, "pet_features"):
                latent_pet = self.ddpm_model.sample(latent_mri.shape[0], latent_mri)
                pet_feats = self.pet_features(latent_pet)
                feats.append(pet_feats)
        if hasattr(self, "tabular_features"):
            tabular_feats = self.tabular_features(tabular)
            feats.append(tabular_feats)
        fused = torch.cat(feats, dim=1)
        out = self.fc(fused)
        return out

    def training_step(self, batch, batch_idx):
        mri = batch['mri']
        labels = batch['label']
        tabular = batch['tabular']

        bs = len(labels)
        outputs = self(mri, tabular)
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
        outputs = self(mri, tabular)
        loss = self.classification_loss(outputs, labels)
        metrics = self.val_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        mri = batch['mri']
        labels = batch['label']
        tabular = batch['tabular']

        bs = len(labels)
        outputs = self(mri, tabular)
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
            eta_min=self.eta_min
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
