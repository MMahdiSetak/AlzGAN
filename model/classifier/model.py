import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, AUROC, Specificity
import monai.transforms as T
import torch.nn.functional as F

from augmentation.random_transform import Compose3D, RandAffine3D
from dataset import log_to_file_image, log_video
from model.classifier.model_blocks import DualCNNBranch, Vision3DTransformer, CrossAttentionFusion
from model.classifier.module import MRICNN, MRI3DViT, AlzheimerCNN3D


class Classifier(pl.LightningModule):
    def __init__(self, image_size=256, patch_size=8, embed_dim=2048, depth=6, heads=16, vit_depth=12, vit_heads=16,
                 lr=1e-3, weight_decay=1e-2, class_weights=None, epochs=500):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        num_classes = 3

        # Model components
        # self.cnn_branch = DualCNNBranch()
        # self.vit_branch = Vision3DTransformer(input_size=(64, 64, 64))
        # self.fusion = CrossAttentionFusion(feature_dim=256)
        # # Final classifier
        # self.classifier = nn.Sequential(
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(64, num_classes)
        # )
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
        self.classifier = MRICNN(num_classes=3, dropout_rate=0.6, channels=32)
        # self.classifier = AlzheimerCNN3D()

        # self.mri_vit = MRI3DViT(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim, depth=vit_depth,
        #                         num_heads=vit_heads)
        # self.gan = Generator()
        # self.pet_proj = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(),
        #     nn.Linear(2048, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(2048, embed_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.mri_proj = nn.Sequential(
        #     nn.AdaptiveAvgPool3d(1),
        #     nn.Flatten(),
        #     nn.Linear(2048, 2048),
        #     nn.BatchNorm1d(2048),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(2048, embed_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True), num_layers=depth
        # )
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.head = nn.Linear(embed_dim, num_classes)

        self.train_transforms = Compose3D([
            RandAffine3D(translate_range=(5, 5, 5), range_x=np.pi / 18, range_y=np.pi / 18, range_z=np.pi / 18,
                         prob=0.5)
        ])

        # self.train_transforms = T.Compose([
        #     T.RandRotate(range_x=np.pi / 18, range_y=np.pi / 18, range_z=np.pi / 18, prob=0.3),
        #     T.Rand3DElastic(sigma_range=(2, 5), magnitude_range=(0.1, 0.3), prob=0.3),
        #     T.RandAffine(translate_range=(10, 10, 10), scale_range=(-0.1, 0.1), prob=0.5),
        #     # T.RandGaussianNoise(std=0.01, prob=0.2),
        #     # T.RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),
        #     T.RandBiasField(prob=0.3)
        # ])

        # def apply_transform(self, mri):
        #     mri = mri.unsqueeze(1).div_(255.0)
        #     transformed_mri = []
        #     for i in range(mri.shape[0]):
        #         sample = mri[i]
        #         transformed_sample = self.train_transforms(sample)
        #         transformed_mri.append(transformed_sample)
        #
        #     mri = torch.stack(transformed_mri, dim=0).multiply_(2).sub_(1)
        #     mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        #     return mri

    def forward(self, mri):
        # mri = mri.multiply_(2).sub_(1)
        # mri = F.interpolate(mri, size=(80, 80, 64), mode='trilinear', align_corners=False)
        # mri = mri.to(torch.float32).div_(255).unsqueeze_(1)
        # mri = self.train_transforms(mri)
        # mri = F.interpolate(mri, size=(64, 64, 64), mode='trilinear', align_corners=False)
        # log_video(mri[0, 0].cpu().numpy(), f"test/mri")
        # log_to_file_image(mri[0, 0].cpu())
        # log_video(mri[0, 0].cpu().numpy(), f"test/trans_mri")
        out = self.classifier(mri)
        # mri_token = self.mri_vit(mri)  # [B, embed_dim]
        # diff_token = self.diffusion_extractor(mri)  # [B, 1, embed_dim]
        # mri_gan_token, pet_gan_token = self.gan(mri)
        # mri_gan_token = self.mri_proj(mri_gan_token)
        # pet_gan_token = self.pet_proj(pet_gan_token)
        # demo_token = self.demo_encoder(demo)  # [B, 1, embed_dim]
        # clinical_token = self.clinical_encoder(clinical)  # [B, 1, embed_dim]
        #
        # tokens = torch.cat([mri_token.unsqueeze(1), mri_gan_token.unsqueeze(1), pet_gan_token.unsqueeze(1)],
        #                    dim=1)  # [B, 4, 768]
        # tokens = mri_token.unsqueeze(1)
        # cls_token = self.cls_token.expand(tokens.size(0), -1, -1)
        # tokens = torch.cat([cls_token, tokens], dim=1)  # [B, 5, 768]
        #
        # fused = self.transformer(tokens)  # [B, 5, 768]
        # fused = fused[:, 0]  # CLS token
        # out = self.head(fused)  # [B, 3]
        # out = self.head(mri_token)  # [B, 3]
        return out

    # def forward(self, x):
    #     # Extract features from both branches
    #     # x_cnn = F.interpolate(x, size=(128, 128, 128), mode='trilinear', align_corners=False)
    #     x_vit = F.interpolate(x, size=(64, 64, 64), mode='trilinear', align_corners=False)
    #     cnn_features = self.cnn_branch(x)  # Original size for CNN
    #     vit_features = self.vit_branch(x_vit)  # Resized for ViT
    #     # Fuse features
    #     fused_features = self.fusion(cnn_features, vit_features)
    #     # Final classification
    #     output = self.classifier(fused_features)
    #     return output

    def training_step(self, batch, batch_idx):
        mri, labels = batch
        mri = mri.to(torch.float32).div_(255).unsqueeze_(1)
        mri = self.train_transforms(mri)
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
        inputs, labels = batch
        inputs = inputs.to(torch.float32).div_(255).unsqueeze_(1)
        # inputs = inputs.unsqueeze(1).div_(127.5).sub_(1)
        # inputs = F.interpolate(inputs, size=(128, 128, 128), mode='trilinear', align_corners=False)
        bs = len(labels)
        outputs = self(inputs)
        loss = self.classification_loss(outputs, labels)
        metrics = self.val_metrics(outputs, labels)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        self.log('loss/val', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = inputs.to(torch.float32).div_(255).unsqueeze_(1)
        # inputs = inputs.unsqueeze(1).div_(127.5).sub_(1)
        # inputs = F.interpolate(inputs, size=(128, 128, 128), mode='trilinear', align_corners=False)
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

        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0)
        # # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler,
        #         'monitor': 'loss/train',
        #         'interval': 'epoch',
        #         'frequency': 1,
        #     }
        # }
