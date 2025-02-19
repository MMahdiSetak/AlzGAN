import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model.dataloader import DataLoader
from model.log import Logger

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)
logger = Logger("diffusion")


class CrossAttention3D(nn.Module):
    """
    A cross-attention block for 3D feature maps.
    The query comes from the PET branch and the keys/values from the MRI condition.
    The input features are reshaped to sequences (flattening spatial dims),
    and nn.MultiheadAttention is applied.
    """

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, cond):
        # query, cond: (B, C, D, H, W)
        B, C, D, H, W = query.shape
        L = D * H * W
        # Flatten spatial dims: shape (B, L, C)
        query_flat = query.view(B, C, -1).permute(0, 2, 1)
        cond_flat = cond.view(B, C, -1).permute(0, 2, 1)
        # Apply multi-head attention: query attends to condition
        attn_output, _ = self.attn(query_flat, cond_flat, cond_flat)
        # Add residual connection and normalize
        attn_output = self.norm(query_flat + attn_output)
        # Optionally project the output
        attn_output = self.proj(attn_output)
        # Reshape back to (B, C, D, H, W)
        out = attn_output.permute(0, 2, 1).view(B, C, D, H, W)
        return out


# ---------------------------
# 2) 3D U-Net Building Blocks
# ---------------------------
class DoubleConv3D(nn.Module):
    """
    Two successive 3D convolutions with BatchNorm and ReLU activation.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------
# 3) 3D U-Net with Attention-based Conditioning
# ---------------------------
class UNet3DConditionedAttention(nn.Module):
    """
    3D U-Net for diffusion where the noisy PET is processed in the main branch,
    and the MRI conditioning is injected via cross-attention at the bottleneck.
    Instead of concatenation, the MRI is first resampled and projected to the same
    feature dimension as the PET bottleneck.

    Inputs:
      - pet: noisy PET volume, shape (B, 1, 100, 140, 96)
      - mri: full MRI volume, shape (B, 1, 160, 200, 180)
    Output:
      - A prediction (e.g. noise prediction) with shape matching the PET target.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=32, num_heads=4):
        super().__init__()
        # Downsampling path for PET branch
        self.conv_down1 = DoubleConv3D(in_channels, base_channels)
        self.pool1 = nn.MaxPool3d(2)
        self.conv_down2 = DoubleConv3D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck (PET branch)
        self.conv_bottleneck = DoubleConv3D(base_channels * 2, base_channels * 4)

        # Conditioning: project MRI to match bottleneck dimensions.
        # We resample MRI to the same spatial size as the bottleneck.
        self.cond_proj = nn.Conv3d(1, base_channels * 4, kernel_size=3, padding=1)
        self.cross_attn = CrossAttention3D(embed_dim=base_channels * 4, num_heads=num_heads)

        # Upsampling path
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(base_channels * 2, base_channels)

        # Final convolution to get the desired output channel.
        self.out_conv = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, pet, mri):
        # pet: (B, 1, 100, 140, 96)
        # mri: (B, 1, 160, 200, 180)

        # Downsampling PET branch
        x1 = self.conv_down1(pet)  # (B, base_channels, 100, 140, 96)
        x1p = self.pool1(x1)  # (B, base_channels, 50, 70, 48)

        x2 = self.conv_down2(x1p)  # (B, base_channels*2, 50, 70, 48)
        x2p = self.pool2(x2)  # (B, base_channels*2, 25, 35, 24)

        xb = self.conv_bottleneck(x2p)  # (B, base_channels*4, 25, 35, 24)

        # Condition: resample MRI to bottleneck resolution and project.
        cond_feat = F.interpolate(mri, size=xb.shape[2:], mode='trilinear', align_corners=False)
        cond_feat = self.cond_proj(cond_feat)  # (B, base_channels*4, 25, 35, 24)

        # Apply cross-attention: PET bottleneck attends to MRI condition.
        xb = xb + self.cross_attn(xb, cond_feat)

        # Upsampling path
        x2u = self.up2(xb)  # (B, base_channels*2, 50, 70, 48)
        x2cat = torch.cat([x2, x2u], dim=1)  # (B, base_channels*2*2, 50, 70, 48)
        x2d = self.conv_up2(x2cat)  # (B, base_channels*2, 50, 70, 48)

        x1u = self.up1(x2d)  # (B, base_channels, 100, 140, 96)
        x1cat = torch.cat([x1, x1u], dim=1)  # (B, base_channels*2, 100, 140, 96)
        x1d = self.conv_up1(x1cat)  # (B, base_channels, 100, 140, 96)

        out = self.out_conv(x1d)  # (B, out_channels, 100, 140, 96)
        return out


# ---------------------------
# 4) Diffusion Helpers (Same as before)
# ---------------------------
def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


class DiffusionConfig:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta = linear_beta_schedule(timesteps, beta_start, beta_end)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)
        self.alpha_bar_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_bar[:-1]], dim=0)


def q_sample(x_0, t, config):
    batch_size = x_0.shape[0]
    device = x_0.device
    alpha_bar_t = config.alpha_bar[t].view(-1, 1, 1, 1, 1).to(device)
    noise = torch.randn_like(x_0)
    return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1. - alpha_bar_t) * noise, noise


# ---------------------------
# 5) Training Loop (Skeleton)
# ---------------------------
def train_diffusion(model, config, data_generator, val_data_generator, epochs=100, batch_size=32,
                    steps_per_epoch=100, steps_per_epoch_val=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train().to(device)

    for epoch in range(epochs):
        epoch_train_loss = 0.0
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", unit="step") as pbar:
            for step in range(steps_per_epoch):
                mri, pet = next(data_generator)

                t_batch = torch.randint(0, config.timesteps, (batch_size,), device=device).long()
                x_t, noise = q_sample(pet, t_batch, config)
                # Use the PET branch only as input; MRI conditions via attention.
                noise_pred = model(x_t, mri)
                loss = F.mse_loss(noise_pred, noise)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():06.4f}"})
                pbar.update(1)
        epoch_train_loss /= steps_per_epoch

        logger.writer.add_scalar("Loss", epoch_train_loss, epoch + 1)
        # psnr = evaluate(generator, val_data_generator, steps_per_epoch_val)
        # logger.writer.add_scalar("PSNR", psnr, epoch + 1)

        # if psnr > best_psnr:
        #     best_psnr = psnr
        #     checkpoint = {
        #         "epoch": epoch + 1,
        #         "generator_state_dict": generator.state_dict(),
        #         "discriminator_state_dict": discriminator.state_dict(),
        #         "optimizer_g_state_dict": optimizer_g.state_dict(),
        #         "optimizer_d_state_dict": optimizer_d.state_dict(),
        #         "best_psnr": best_psnr,
        #     }
        #     torch.save(checkpoint, os.path.join(logger.log_dir, "gan.pth"))
        #     print(f"Saved model with PSNR: {psnr:.2f}")
        # visualize_progress(fake_pet[0][0].cpu().detach().numpy(), f"{epoch:03d}_fake_pet")
        # visualize_progress(real_mri[0][0].cpu().detach().numpy(), f"{epoch:03d}_real_mri")
        # visualize_progress(real_pet[0][0].cpu().detach().numpy(), f"{epoch:03d}_real_pet")
    print("Training Complete!")


batch_size = 1
data_loader = DataLoader('dataset/mri_pet_label_v3.hdf5', batch_size)
train_data_generator = data_loader.data_generator(batch_size, "train", pet=True, label=False)
val_data_generator = data_loader.data_generator(batch_size, "val", pet=True, label=False)
# Create model with attention-based conditioning.
model = UNet3DConditionedAttention(in_channels=1, out_channels=1, base_channels=16, num_heads=4)

# Diffusion configuration (e.g., 200 timesteps for a demo)
diffusion_config = DiffusionConfig(timesteps=200, beta_start=1e-4, beta_end=0.02)

train_diffusion(model, diffusion_config, train_data_generator, val_data_generator, epochs=500, batch_size=batch_size,
                steps_per_epoch=data_loader.steps_per_epoch_train,
                steps_per_epoch_val=data_loader.steps_per_epoch_train)
