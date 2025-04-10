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
        # self.labels = extract_patches()
        # self.lb_0 = VoxelFCN(3818095)
        # self.lb_2 = VoxelFCN(299661)
        # self.lb_3 = VoxelFCN(316806)
        # self.lb_4 = VoxelFCN(10579)
        # self.lb_5 = VoxelFCN(641)
        # self.lb_7 = VoxelFCN(19090)
        # self.lb_8 = VoxelFCN(66771)
        # self.lb_10 = VoxelFCN(9773)
        # self.lb_11 = VoxelFCN(5650)
        # self.lb_12 = VoxelFCN(7200)
        # self.lb_13 = VoxelFCN(2234)
        # self.lb_14 = VoxelFCN(1106)
        # self.lb_15 = VoxelFCN(2903)
        # self.lb_16 = VoxelFCN(28652)
        # self.lb_17 = VoxelFCN(5815)
        # self.lb_18 = VoxelFCN(2283)
        # self.lb_24 = VoxelFCN(401470)
        # self.lb_26 = VoxelFCN(887)
        # self.lb_28 = VoxelFCN(5983)
        # self.lb_41 = VoxelFCN(299032)
        # self.lb_42 = VoxelFCN(317486)
        # self.lb_43 = VoxelFCN(10508)
        # self.lb_44 = VoxelFCN(601)
        # self.lb_46 = VoxelFCN(19502)
        # self.lb_47 = VoxelFCN(67542)
        # self.lb_49 = VoxelFCN(9764)
        # self.lb_50 = VoxelFCN(5659)
        # self.lb_51 = VoxelFCN(7192)
        # self.lb_52 = VoxelFCN(2197)
        # self.lb_53 = VoxelFCN(5851)
        # self.lb_54 = VoxelFCN(2317)
        # self.lb_58 = VoxelFCN(893)
        # self.lb_60 = VoxelFCN(5857)

    def forward(self, image):
        out = [[] for _ in range(33)]
        for label in self.labels:
            out.append(process_voxels(image, self.lb_fcn[label], label))
        # out[0] = self.lb_0(process_voxels(image, self.labels, 0))
        # out[2] = self.lb_2(process_voxels(image, self.labels, 2))
        # TODO write all the 33 lines like above
        return out
