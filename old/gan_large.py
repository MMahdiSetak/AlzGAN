import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from tqdm import tqdm

from model.dataloader import DataLoader
from model.log import Logger

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
logger = Logger("gan")
mri_target_shape = (1, 160, 200, 180)
pet_target_shape = (1, 100, 140, 96)


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
        )

        self.bottleneck = nn.Sequential(
            nn.Conv3d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),

            nn.Conv3d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),

            nn.Conv3d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout3d(0.2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),

            nn.ConvTranspose3d(128, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x[:, :, :100, :140, :96]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(1024, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


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


generator = Generator().to(device)
discriminator = Discriminator().to(device)
# summary(discriminator, pet_target_shape, batch_size=1)
summary(generator, mri_target_shape, batch_size=16)

pixelwise_loss = nn.MSELoss()

optimizer_g = optim.Adam(generator.parameters(), lr=0.00003, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=0.00003, betas=(0.5, 0.999))
scheduler = ReduceLROnPlateau(optimizer_g, 'min', patience=3, factor=0.5)

criterion = nn.BCELoss()


def visualize_progress(img, file_name):
    center_slices = [dim // 2 for dim in img.shape]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Axial', 'Coronal', 'Sagittal']

    slices = [img[center_slices[0], :, :], img[:, center_slices[1], :], img[:, :, center_slices[2]]]

    for ax, slice_img, title in zip(axes, slices, titles):
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(title)

    model_dir = f"{logger.log_dir}/model"
    os.makedirs(model_dir, exist_ok=True)
    plt.savefig(f"{model_dir}/{file_name}.png")
    plt.close(fig)


def train_gan(generator, discriminator, data_generator, val_data_generator, epochs=100, batch_size=32,
              steps_per_epoch=100, steps_per_epoch_val=50):
    valid = torch.ones((batch_size, *discriminator(torch.randn(1, *pet_target_shape).to(device)).shape[1:])).to(device)
    fake = torch.zeros((batch_size, *discriminator(torch.randn(1, *pet_target_shape).to(device)).shape[1:])).to(device)

    best_psnr = -float('inf')  # Initialize best PSNR to negative infinity

    for epoch in range(epochs):
        epoch_train_gloss = 0
        epoch_train_dloss = 0
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", unit="step") as pbar:
            for step in range(steps_per_epoch):
                real_mri, real_pet = next(data_generator)
                fake_pet = generator(real_mri)

                optimizer_d.zero_grad()
                real_loss = criterion(discriminator(real_pet), valid)
                fake_loss = criterion(discriminator(fake_pet.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                g_loss = criterion(discriminator(fake_pet), valid) + pixelwise_loss(fake_pet, real_pet)
                g_loss.backward()
                optimizer_g.step()

                epoch_train_gloss += g_loss.item()
                epoch_train_dloss += d_loss.item()

                pbar.set_postfix({"D_Loss": f"{d_loss.item():06.4f}", "G_Loss": f"{g_loss.item():07.4f}"})
                pbar.update(1)
        epoch_train_gloss /= steps_per_epoch
        epoch_train_dloss /= steps_per_epoch

        logger.writer.add_scalar("D_Loss", epoch_train_dloss, epoch + 1)
        logger.writer.add_scalar("G_Loss", epoch_train_gloss, epoch + 1)
        psnr = evaluate(generator, val_data_generator, steps_per_epoch_val)
        logger.writer.add_scalar("PSNR", psnr, epoch + 1)

        if psnr > best_psnr:
            best_psnr = psnr
            checkpoint = {
                "epoch": epoch + 1,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "optimizer_g_state_dict": optimizer_g.state_dict(),
                "optimizer_d_state_dict": optimizer_d.state_dict(),
                "best_psnr": best_psnr,
            }
            torch.save(checkpoint, os.path.join(logger.log_dir, "gan.pth"))
            print(f"Saved model with PSNR: {psnr:.2f}")
        visualize_progress(fake_pet[0][0].cpu().detach().numpy(), f"{epoch:03d}_fake_pet")
        visualize_progress(real_mri[0][0].cpu().detach().numpy(), f"{epoch:03d}_real_mri")
        visualize_progress(real_pet[0][0].cpu().detach().numpy(), f"{epoch:03d}_real_pet")


batch_size = 16
data_loader = DataLoader('dataset/mri_pet_label.hdf5', batch_size)

logger.save_model_metadata(generator, mri_target_shape, "generator", batch_size)
logger.save_model_metadata(discriminator, pet_target_shape, "discriminator", batch_size)

train_data_generator = data_loader.data_generator(batch_size, "train", pet=True, label=False)
val_data_generator = data_loader.data_generator(batch_size, "val", pet=True, label=False)
train_gan(generator, discriminator, train_data_generator, val_data_generator, epochs=500, batch_size=batch_size,
          steps_per_epoch=data_loader.steps_per_epoch_train, steps_per_epoch_val=data_loader.steps_per_epoch_train)
