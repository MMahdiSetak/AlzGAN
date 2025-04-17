import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, Specificity

from seg.patch import get_patch_indices

torch.set_float32_matmul_precision('medium')


class VoxelFCN(nn.Module):
    def __init__(self, input_size, output_size=128, dropout=0.2, norm='batch'):
        super(VoxelFCN, self).__init__()
        # l1_size = max(output_size * 4, input_size // 2)
        # l2_size = max(output_size * 2, input_size // 4)
        l1_size = output_size * 4
        l2_size = output_size * 2
        if norm == 'batch':
            l1_norm = nn.BatchNorm1d(l1_size)
            l2_norm = nn.BatchNorm1d(l2_size)
        elif norm == 'instance':
            l1_norm = nn.InstanceNorm1d(l1_size)
            l2_norm = nn.InstanceNorm1d(l2_size)
        else:
            l1_norm = nn.LayerNorm(l1_size)
            l2_norm = nn.LayerNorm(l2_size)
        self.fc = nn.Sequential(
            nn.Linear(input_size, l1_size),
            nn.ReLU(),
            l1_norm,
            nn.Dropout(dropout),
            nn.Linear(l1_size, l2_size),
            nn.ReLU(),
            l2_norm,
            nn.Dropout(dropout),
            nn.Linear(l2_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class SegmentTransformer(pl.LightningModule):
    def __init__(self, embedding_size=128, dropout=0.2, lr=1e-3, norm='batch'):
        super(SegmentTransformer, self).__init__()
        self.lr = lr
        self.classification_loss = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=3)
        metrics = MetricCollection({
            "accuracy": Accuracy(task="multiclass", num_classes=3),
            "precision": Precision(task="multiclass", num_classes=3),
            "recall": Recall(task="multiclass", num_classes=3),
            "f1_score": F1Score(task="multiclass", num_classes=3),
            "auc": AUROC(task="multiclass", num_classes=3),
            "specificity": Specificity(task="multiclass", num_classes=3),
        })
        self.metrics = metrics
        self.labels = [
            2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52,
            53, 54, 58, 60
        ]
        self.patches = get_patch_indices()
        self.lb_fcn = nn.ModuleDict({
            str(label): VoxelFCN(input_size=self.patches[label].sum(), output_size=embedding_size, dropout=dropout,
                                 norm=norm)
            for label in self.labels
        })

        # Transformer-related layers
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=4,  # Number of attention heads
            dim_feedforward=512,  # Feedforward layer size
            dropout=0.2,  # Dropout to prevent overfitting
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
        out = [self.lb_fcn[str(label)](image[:, self.patches[label]]) for label in self.labels]
        transformer_input = torch.stack(out, dim=1)
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
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=False)
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

        # return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        # return optimizer


def test():
    image = torch.randn((2, 160, 200, 180))
    model = SegmentTransformer()
    outputs = model(image)
    for label, out_tensor in outputs.items():
        print(f"Label {label}: output shape {out_tensor.shape}")
