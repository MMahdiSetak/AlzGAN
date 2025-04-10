import torch
import torch.nn as nn
import torch.optim as optim

from seg.patch import get_patch_indices


class VoxelFCN(nn.Module):
    def __init__(self, input_size, output_size=128):
        super(VoxelFCN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        return self.fc(x)


class SegmentFCN(nn.Module):
    def __init__(self):
        super(SegmentFCN, self).__init__()
        self.labels = [
            0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52,
            53, 54, 58, 60
        ]
        self.patches = get_patch_indices()
        self.lb_fcn = nn.ModuleDict({
            str(label): VoxelFCN(input_size=self.patches[label].sum())
            for label in self.labels
        })

        # Transformer-related layers
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,  # This should match the size of the output of VoxelFCN
            nhead=4,  # Number of attention heads
            dim_feedforward=512,  # Feedforward layer size
            dropout=0.1  # Dropout to prevent overfitting
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=4
        )

        # Final classification output layer (3 classes)
        self.fc_out = nn.Linear(128, 3)  # 3 output classes for classification

    def forward(self, image):
        out = {}
        for label in self.labels:
            indices = self.patches[label]
            voxels = image[:, indices]
            out[label] = self.lb_fcn[str(label)](voxels)

        # Convert the outputs from the 33 labels into a single tensor for Transformer processing
        # Each label has a 128-dimensional output (we are treating this as a sequence of 33 tokens)
        transformer_input = torch.stack([out[label] for label in self.labels], dim=1)  # Shape: (batch_size, 33, 128)

        # Transformer requires (seq_len, batch_size, d_model), so we need to permute the dimensions
        transformer_input = transformer_input.permute(1, 0, 2)  # Shape: (33, batch_size, 128)

        # Pass through the transformer encoder
        transformer_output = self.transformer_encoder(transformer_input)

        # Optionally, you can pool the transformer output (e.g., mean, sum, etc.)
        transformer_output = transformer_output.mean(dim=0)  # Shape: (batch_size, 128)

        # Pass the transformer output through the final output layer for classification (3 classes)
        final_output = self.fc_out(transformer_output)  # Shape: (batch_size, 3)

        return final_output


def run():
    image = torch.randn((2, 160, 200, 180))
    model = SegmentFCN()
    outputs = model(image)
    for label, out_tensor in outputs.items():
        print(f"Label {label}: output shape {out_tensor.shape}")
