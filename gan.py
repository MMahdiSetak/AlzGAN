import math
import os
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Dimensions of the input and output images
mri_target_shape = (1, 80, 96, 96)  # (channels, depth, height, width)
pet_target_shape = (1, 35, 64, 64)


# Data loader function  
def pair_data_generator(hdf5_file, batch_size, split):
    with h5py.File(hdf5_file, 'r') as file:
        # mri_images = file['mri']
        # pet_images = file['pet']
        if split == 'train':
            mri_images = file['mri_train']
            pet_images = file['pet_train']
            labels = file['label_train']
        elif split == 'val':
            mri_images = file['mri_val']
            pet_images = file['pet_val']
            labels = file['label_val']
        else:
            mri_images = file['mri_test']
            pet_images = file['pet_test']
            labels = file['label_test']
        n = len(labels)
        indices = np.arange(n)
        while True:
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                batch_indices = indices[i:end]  # Get batch indices

                if len(batch_indices) < batch_size:
                    # Randomly sample additional indices to pad the batch
                    r = np.random.randint(0, n - batch_size)
                    additional_indices = indices[r:r + (batch_size - len(batch_indices))]
                    # additional_indices = np.random.choice(indices, batch_size - len(batch_indices), replace=False)
                    batch_indices = np.concatenate((additional_indices, batch_indices))

                # Retrieve data for the batch
                batch_mri = mri_images[batch_indices]
                batch_pet = pet_images[batch_indices]
                batch_labels = labels[batch_indices]

                batch_mri = torch.Tensor(batch_mri / 256).unsqueeze(1).to(device)  # Add channel dimension
                batch_pet = torch.Tensor(batch_pet / 256).unsqueeze(1).to(device)
                yield batch_mri, batch_pet


# Define the Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),

            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),

            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Tanh to get output in range [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x[:, :, :35, :64, :64]


# Define the Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 1, kernel_size=4, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Conv3d(1, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm3d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv3d(256, 1, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         return self.model(x).view(-1, 1).squeeze(1)  # Flatten to (batch, 1)

def calculate_psnr(fake_pet, real_pet):
    mse_loss = nn.MSELoss()(fake_pet, real_pet)
    if mse_loss == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
    return psnr.item()


def evaluate(generator, val_data_generator, steps_per_epoch_val):
    generator.eval()
    total_psnr = 0
    with torch.no_grad():
        for _ in range(steps_per_epoch_val):
            mri, pet = next(val_data_generator)
            fake_pet = generator(mri)
            total_psnr += calculate_psnr(fake_pet, pet)
    generator.train()  # Switch back to training mode
    return total_psnr / steps_per_epoch_val


# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and Optimizers
# adversarial_loss = nn.BCELoss()
pixelwise_loss = nn.MSELoss()

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# TensorBoard writer
time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
log_dir = os.path.join("log/gan", time_stamp)
writer = SummaryWriter(log_dir=log_dir)


# Training function
def visualize_progress(img, file_name):
    center_slices = [dim // 2 for dim in img.shape]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Axial', 'Coronal', 'Sagittal']

    slices = [img[center_slices[0], :, :], img[:, center_slices[1], :], img[:, :, center_slices[2]]]

    for ax, slice_img, title in zip(axes, slices, titles):
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    # Save the figure to a file
    plt.savefig(f"model/GAN/{time_stamp}/{file_name}.png")
    plt.close(fig)


def train_gan(generator, discriminator, data_generator, val_data_generator, epochs=100, batch_size=32,
              steps_per_epoch=100, steps_per_epoch_val=50):
    valid = torch.ones((batch_size, *discriminator(torch.randn(1, *pet_target_shape).to(device)).shape[1:])).to(device)
    fake = torch.zeros((batch_size, *discriminator(torch.randn(1, *pet_target_shape).to(device)).shape[1:])).to(device)
    model_path = f"model/GAN/{time_stamp}"
    os.makedirs(model_path, exist_ok=True)

    best_psnr = -float('inf')  # Initialize best PSNR to negative infinity

    for epoch in range(epochs):
        epoch_train_gloss = 0
        epoch_train_dloss = 0
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", unit="step") as pbar:
            for step in range(steps_per_epoch):
                real_mri, real_pet = next(data_generator)
                # Generate fake PET images
                fake_pet = generator(real_mri)
                # visualize_progress(fake_pet[0][0].cpu().detach().numpy())
                # visualize_progress(real_pet[0][0].cpu().detach().numpy())
                # visualize_progress(real_mri[0][0].cpu().detach().numpy())

                # Train Discriminator
                optimizer_d.zero_grad()
                real_loss = criterion(discriminator(real_pet), valid)
                fake_loss = criterion(discriminator(fake_pet.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                g_loss = criterion(discriminator(fake_pet), valid) + pixelwise_loss(fake_pet, real_pet)
                g_loss.backward()
                optimizer_g.step()

                epoch_train_gloss += g_loss.item()
                epoch_train_dloss += d_loss.item()
                # Log progress

                # Update progress bar
                pbar.set_postfix({"D_Loss": f"{d_loss.item():06.4f}", "G_Loss": f"{g_loss.item():07.4f}"})
                pbar.update(1)
        # After each epoch, evaluate on validation set
        epoch_train_gloss /= steps_per_epoch
        epoch_train_dloss /= steps_per_epoch

        writer.add_scalar("D_Loss", epoch_train_dloss, epoch + 1)
        writer.add_scalar("G_Loss", epoch_train_gloss, epoch + 1)
        psnr = evaluate(generator, val_data_generator, steps_per_epoch_val)
        writer.add_scalar("PSNR", psnr, epoch + 1)

        # Save the model if it achieves the best PSNR
        if psnr > best_psnr:
            best_psnr = psnr
            torch.save(generator.state_dict(), f"{model_path}/best_generator_epoch_{epoch + 1}_psnr_{psnr:.2f}.pth")
            print(f"Saved model with PSNR: {psnr:.2f}")
        visualize_progress(fake_pet[5][0].cpu().detach().numpy(), f"{epoch:03d}_fake_pet")
        visualize_progress(real_mri[5][0].cpu().detach().numpy(), f"{epoch:03d}_real_mri")
        visualize_progress(real_pet[5][0].cpu().detach().numpy(), f"{epoch:03d}_real_pet")


# Example usage
hdf5_file = 'dataset/mri_pet_label_small_v2.hdf5'
with h5py.File(hdf5_file, 'r') as file:
    train_size = len(file['label_train'])
    val_size = len(file['label_val'])
    test_size = len(file['label_test'])
batch_size = 12
# train_size = 2124
# train_size = 983
# val_size = 266
steps_per_epoch = math.ceil(train_size / batch_size)
steps_per_epoch_val = math.ceil(val_size / batch_size)
train_data_generator = pair_data_generator(hdf5_file, batch_size, "train")
val_data_generator = pair_data_generator(hdf5_file, batch_size, "val")
train_gan(generator, discriminator, train_data_generator, val_data_generator, epochs=300, batch_size=batch_size,
          steps_per_epoch=steps_per_epoch, steps_per_epoch_val=steps_per_epoch_val)
