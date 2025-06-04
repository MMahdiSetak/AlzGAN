import torch
import torch.nn as nn


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
