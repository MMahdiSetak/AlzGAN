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
