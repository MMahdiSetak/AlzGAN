import torch
import torch.nn as nn
import math


class DualCNNBranch(nn.Module):
    """Enhanced Dual CNN Branch based on El-Assy et al. 2024"""

    def __init__(self, input_channels=1, dropout_rate=0.2):
        super().__init__()

        # CNN Branch 1: 3x3 filters
        self.cnn1 = nn.Sequential(
            # First block
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Second block
            nn.Conv3d(16, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Third block
            nn.Conv3d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(dropout_rate)
        )

        # CNN Branch 2: 5x5 filters
        self.cnn2 = nn.Sequential(
            # First block
            nn.Conv3d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Second block
            nn.Conv3d(32, 128, kernel_size=5, padding=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Third block
            nn.Conv3d(128, 512, kernel_size=5, padding=2),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Dropout3d(dropout_rate)
        )

        # Adaptive pooling for variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Feature projection layers
        self.proj1 = nn.Linear(256, 128)
        self.proj2 = nn.Linear(512, 128)

    def forward(self, x):
        # CNN Branch 1
        features1 = self.cnn1(x)
        features1 = self.adaptive_pool(features1)
        features1 = features1.view(features1.size(0), -1)
        features1 = self.proj1(features1)

        # CNN Branch 2
        features2 = self.cnn2(x)
        features2 = self.adaptive_pool(features2)
        features2 = features2.view(features2.size(0), -1)
        features2 = self.proj2(features2)

        # Concatenate features
        combined_features = torch.cat([features1, features2], dim=1)
        return combined_features


class PositionalEncoding3D(nn.Module):
    """3D Positional Encoding for Vision Transformer"""

    def __init__(self, embed_dim, depth=64, height=64, width=64):
        super().__init__()
        self.embed_dim = embed_dim

        # Create 3D positional encoding
        pe = torch.zeros(depth * height * width, embed_dim)
        position = torch.arange(0, depth * height * width).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() *
                             -(math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class Vision3DTransformer(nn.Module):
    """3D Vision Transformer for brain MRI"""

    def __init__(self, input_size=(64, 64, 64), patch_size=(8, 8, 8),
                 embed_dim=768, num_heads=12, num_layers=6, num_classes=3):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (input_size[0] // patch_size[0]) * \
                           (input_size[1] // patch_size[1]) * \
                           (input_size[2] // patch_size[2])

        # Patch embedding
        self.patch_embed = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding
        self.pos_encoding = PositionalEncoding3D(embed_dim,
                                                 input_size[0] // patch_size[0],
                                                 input_size[1] // patch_size[1],
                                                 input_size[2] // patch_size[2])

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 256)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)
        x = self.ln(x)
        x = self.head(x)

        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between CNN and ViT features"""

    def __init__(self, feature_dim=128, num_heads=8):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim)
        )

    def forward(self, cnn_features, vit_features):
        # Cross-attention: CNN queries ViT
        cnn_features = cnn_features.unsqueeze(1)  # Add sequence dimension
        vit_features = vit_features.unsqueeze(1)

        attn_output, _ = self.cross_attention(cnn_features, vit_features, vit_features)
        cnn_enhanced = self.norm1(cnn_features + attn_output)

        # Feed-forward
        ffn_output = self.ffn(cnn_enhanced)
        output = self.norm2(cnn_enhanced + ffn_output)

        return output.squeeze(1)
