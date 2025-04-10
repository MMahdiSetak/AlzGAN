import torch
import torch.nn as nn
import torch.optim as optim

from seg.patch import extract_patches


class VoxelFCN(nn.Module):
    def __init__(self, input_size, output_size=128):
        super(VoxelFCN, self).__init__()
        # Define the fully connected layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation
        x = torch.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)  # Output layer
        return x


def process_voxels(image, labels, label_id):
    voxel_indices = (labels == label_id)
    voxels = image[voxel_indices]
    return voxels


class SegmentFCN(nn.Module):
    def __init__(self):
        super(SegmentFCN, self).__init__()
        self.labels = [
            0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52,
            53, 54, 58, 60
        ]
        self.lb_fcn = {}
        self.patches = extract_patches()
        for label in self.labels:
            self.lb_fcn[label] = VoxelFCN(len(self.patches[label]))

    def forward(self, image):
        out = [[] for _ in range(33)]
        for label in self.labels:
            out.append(process_voxels(image, self.lb_fcn[label], label))
        return out
