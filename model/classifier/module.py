import torch
import torch.nn as nn
import torchvision.models as models


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


class Hybrid3DViT(nn.Module):
    def __init__(self, image_size=128, embed_dim=2048, depth=24, num_heads=16):
        super().__init__()
        # 3D CNN backbone using ResNet50 adapted for 3D (placeholder; replace with actual 3D ResNet50)
        self.cnn = models.video.r3d_18(weights=None)  # Using r3d_18 as a proxy; ideally use 3D ResNet50
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove avgpool and fc layers

        # Note: For input [B, 1, 128, 128, 128], output is [B, 512, 8, 8, 8] with r3d_18.
        # With a true 3D ResNet50, expect [B, 2048, 8, 8, 8]. Adjust embed_dim accordingly.
        num_patches = (image_size // 16) ** 3  # 8^3 = 512 patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Projection layer to match embed_dim if CNN output channels differ
        self.proj = nn.Linear(512, embed_dim)  # Adjust based on actual CNN output channels (2048 for ResNet50)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True),
            num_layers=depth
        )
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x):
        x = self.cnn(x)  # [B, 512, 8, 8, 8] with r3d_18; [B, 2048, 8, 8, 8] with ResNet50
        x = x.flatten(2).transpose(1, 2)  # [B, 512, 512] or [B, 512, 2048]
        x = self.proj(x)  # [B, 512, 2048] if needed
        x = x + self.pos_embed  # Add positional embeddings
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, 2048]
        x = torch.cat([cls_token, x], dim=1)  # [B, 513, 2048]
        x = self.transformer(x)  # [B, 513, 2048]
        x = x[:, 0]  # CLS token: [B, 2048]
        out = self.head(x)  # [B, 3]
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.Conv3d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.Conv3d(1024, 2048, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv3d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),

            nn.Conv3d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),

            nn.Conv3d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),
        )

    def forward(self, x):
        enc = self.encoder(x)
        bot = self.bottleneck(enc)
        return enc, bot


class MRICNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3, channels=64):
        super(MRICNN, self).__init__()

        self.model = nn.Sequential(
            # CNN blocks
            nn.Conv3d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            nn.Conv3d(channels * 4, channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # nn.Conv3d(channels * 8, channels * 16, kernel_size=3, padding=1),
            # nn.BatchNorm3d(channels * 16),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=2, stride=2),

            # Global average pooling and flatten
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),

            # Fully connected layers
            # nn.Linear(channels * 16, channels * 8),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            nn.Linear(channels * 8, channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(channels * 4, num_classes),
        )

    def forward(self, x):
        return self.model(x)
