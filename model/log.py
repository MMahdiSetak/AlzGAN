import io
import os
import sys
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary


class Logger:
    def __init__(self, model_name):
        self.time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.log_dir = os.path.join(f"log/{model_name}", self.time_stamp)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.mri_target_shape = (1, 160, 200, 180)
        self.pet_target_shape = (1, 100, 140, 96)

    def save_model_metadata(self, model, input, name, batch_size):
        with open(f"{self.log_dir}/{name}.txt", "w") as f:
            buffer = io.StringIO()
            sys.stdout = buffer
            summary(model, input, batch_size=batch_size)
            sys.stdout = sys.__stdout__
            out = buffer.getvalue()
            f.write(out)


def log_3d(img, title="", file_name='test'):
    center_slices = [dim // 2 for dim in img.shape]
    # img = np.transpose(img, (0, 2, 1))

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    # titles = ['Axial', 'Coronal', 'Sagittal']

    slices = [
        img[center_slices[0], :, :],
        img[:, center_slices[1], :],
        img[:, :, center_slices[2]]
    ]

    for ax, slice_img in zip(axes, slices):
        rotated_slice = np.rot90(slice_img, k=1)
        ax.imshow(rotated_slice, cmap='gray')
        # ax.set_title(title)
        # ax.axis('off')

    # Save the figure to a file
    plt.show()
    # plt.savefig(f"log/mri/{file_name}.png")
    plt.close(fig)
