import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121


# class SwinUNETRClassifier(nn.Module):
#     def __init__(self, img_size=128, num_classes=3, feature_size=48):
#         super().__init__()
#         self.backbone = SwinTransformer(
#             in_chans=1,
#             img_size=(img_size, img_size, img_size),
#             patch_size=(4, 4, 4),  # or (2,2,2) depending on model
#             window_size=(7, 7, 7),  # or as needed
#             embed_dim=96,
#             depths=(2, 2, 6, 2),
#             num_heads=(3, 6, 12, 24),
#             num_classes=num_classes,
#             dropout_path_rate=0.2,
#         )
#
#     def forward(self, x):
#         return self.backbone(x)


class DenseNet3DClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # spatial_dims=3 → 3D; in_channels=1 → one MRI channel; out_channels=num_classes → classification
        self.model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            block_config=(6, 12, 24, 16),  # default for 121
        )

    def forward(self, x):
        return self.model(x)


class MRI3DViT(nn.Module):
    def __init__(self, image_size=128, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        patch_number = (image_size // patch_size) ** 3
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_number + 1, embed_dim))  # patches + CLS
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, num_heads, batch_first=True), num_layers=depth
        )

    def forward(self, x):
        x = self.patch_embed(x)  # [B, embed_dim, 8, 8, 8]
        x = x.flatten(2).transpose(1, 2)  # [B, patch_number, embed_dim]
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)  # [B, patch_number+1, embed_dim]
        x = x + self.pos_embed
        x = self.transformer(x)
        return x[:, 0]  # CLS token
