import torch.nn as nn
from monai.networks.nets import SwinUNETR


class SwinUNETRClassifier(nn.Module):
    def __init__(self, img_size=128, num_classes=3, feature_size=48):
        super().__init__()
        self.backbone = SwinUNETR(
            img_size=(img_size, img_size, img_size),
            in_channels=1,
            out_channels=feature_size,
            feature_size=feature_size,
            use_checkpoint=True,  # Reduce VRAM, enables deeper nets
        )
        # Use only encoder features for classification
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        # SwinUNETR returns segmentation output, encoder features
        # We'll grab the deepest encoder features
        # Output is dict, use 'encoder_outputs'[-1] for deepest layer
        _, encoder_outputs = self.backbone(x, return_encoder=True)
        features = encoder_outputs[-1]  # (B, C, D, H, W)
        pooled = self.global_pool(features).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)
        return self.fc(pooled)
