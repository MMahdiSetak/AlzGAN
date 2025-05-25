import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, Specificity

from model.diff_class.unet import create_model


class DiffClass(pl.LightningModule):
    def __init__(self, embedding_size=128, dropout=0.2, lr=1e-3):
        super(DiffClass, self).__init__()
        self.lr = lr
        self.classification_loss = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=3),
            "precision": Precision(task="multiclass", num_classes=3),
            "recall": Recall(task="multiclass", num_classes=3),
            "f1_score": F1Score(task="multiclass", num_classes=3),
            "auc": AUROC(task="multiclass", num_classes=3),
            "specificity": Specificity(task="multiclass", num_classes=3),
        })

        self.unet = create_model(128, 128, 1, in_channels=2, out_channels=1)
        # self.mri_proj = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(unet_size, embedding_size),
        #     nn.ReLU(),
        # )
        # self.pet_proj = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(unet_size, embedding_size),
        #     nn.ReLU(),
        # )
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=4
        )
        self.fc_out = nn.Sequential(
            nn.Linear(embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, image):
        noise = torch.randn_like(image)
        mri_features, pet_features = self.unet(torch.cat([noise, image]), 0)
        print("mri: ", mri_features.shape)
        print("pet: ", pet_features.shape)
        # mri_features = self.mri_proj(mri_features)
        # pet_features = self.pet_proj(pet_features)
        transformer_input = torch.stack([mri_features, pet_features], dim=1)
        transformer_output = self.transformer_encoder(transformer_input)
        transformer_output = transformer_output.mean(dim=1)
        final_output = self.fc_out(transformer_output)
        return final_output

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        bs = len(labels)
        outputs = self(inputs)
        loss = self.classification_loss(outputs, labels)

        acc = self.train_accuracy(outputs, labels)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log('train_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        bs = len(labels)
        outputs = self(inputs)
        loss = self.classification_loss(outputs, labels)

        acc = self.val_accuracy(outputs, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        bs = len(labels)
        outputs = self(inputs)

        metrics = self.metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)

        return metrics

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
