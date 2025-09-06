import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MeanSquaredError, MeanAbsoluteError, MetricCollection
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class VectorQuantizer(nn.Module):
    def __init__(self, n_codes, embedding_dim, beta=0.25):
        super().__init__()
        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_codes, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_codes, 1.0 / self.n_codes)

    def forward(self, z):
        # z: B, C, T, H, W (assuming 3D latent)
        # Flatten spatial dimensions
        z_flattened = z.permute(0, 2, 3, 4, 1).reshape(-1, self.embedding_dim)  # (B*T*H*W, C)

        # Compute distances
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find closest encodings
        encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(encoding_indices).view(z.shape)

        # Compute VQ loss
        commitment_loss = torch.mean((z_q.detach() - z)**2)
        codebook_loss = torch.mean((z_q - z.detach())**2)
        vq_loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, vq_loss, encoding_indices


def fix_image_range(x):
    x -= x.min()
    x /= x.max()
    return x


class VQVAE(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.embedding_dim
        self.n_codes = cfg.n_codes
        # self.beta = cfg.beta
        self.beta = 0.25

        self.encoder = Encoder(cfg.image_type, cfg.n_hiddens, cfg.image_channels, cfg.norm_type, cfg.num_groups)
        self.decoder = Decoder(cfg.image_type, cfg.n_hiddens, cfg.image_channels, cfg.norm_type, cfg.num_groups)
        # self.quantizer = VectorQuantizer(self.n_codes, self.embedding_dim, self.beta)

        self.save_hyperparameters()

        metrics = {
            "MAE": MeanAbsoluteError(),
            "MSE": MeanSquaredError(),
            "PSNR": PeakSignalNoiseRatio(data_range=1),
            "SSIM": StructuralSimilarityIndexMeasure(data_range=1)
        }
        self.train_metrics = MetricCollection(metrics, prefix="train/")
        self.val_metrics = MetricCollection(metrics, prefix="val/")

    # def configure_model(self):
    #     """Called before fit/validate/test/predict"""
    #     if hasattr(torch, 'compile'):
    #         self.encoder = torch.compile(self.encoder, mode='reduce-overhead')
    #         self.decoder = torch.compile(self.decoder, mode='reduce-overhead')

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        recon_loss = F.l1_loss(x_recon, x)
        return recon_loss, x_recon, z

    # def forward(self, x):
    #     z = self.encoder(x)
    #     # z = self.pre_quant_conv(z)
    #     z_q, vq_loss, _ = self.quantizer(z)
    #     # z_q = self.post_quant_conv(z_q)
    #     x_recon = self.decoder(z_q)
    #     recon_loss = F.l1_loss(x_recon, x)
    #     total_loss = recon_loss + vq_loss
    #     return total_loss, recon_loss, vq_loss, x_recon, z_q

    def training_step(self, batch, batch_idx):
        bs = batch.shape[0]
        recon_loss, x_recon, _ = self.forward(batch)
        # total_loss, recon_loss, vq_loss, x_recon, _ = self.forward(batch)
        if batch_idx == 0:
            original = fix_image_range(batch)
            x_recon = fix_image_range(x_recon)
            metrics = self.train_metrics(x_recon, original)
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=bs, sync_dist=True)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('train/vq_loss', vq_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return recon_loss

    def validation_step(self, batch, batch_idx):
        bs = batch.shape[0]
        recon_loss, x_recon, vq_output = self.forward(batch)
        # total_loss, recon_loss, vq_loss, x_recon, _ = self.forward(batch)
        original = fix_image_range(batch)
        x_recon = fix_image_range(x_recon)
        metrics = self.val_metrics(x_recon, original)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs, sync_dist=True)
        self.log('val/recon_loss', recon_loss, prog_bar=False, sync_dist=True)
        # self.log('val/vq_loss', vq_loss, prog_bar=False, sync_dist=True)
        return recon_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            betas=(0.5, 0.9),
            fused=True
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.max_epochs,  # Total epochs
            eta_min=self.cfg.eta_min
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            }
        }

    @torch.no_grad()
    def inference(self, x):
        """Separate inference method for logging - not compiled"""
        B, C, T, H, W = x.shape

        # Clone inputs to avoid CUDA graph conflicts
        x = x.clone()

        # Use eval mode temporarily
        was_training = self.training
        self.eval()

        try:
            z = self.encoder(x)
            # z_q, _, _ = self.quantizer(z)
            x_recon = self.decoder(z)

            # Random frame selection for logging
            frame_idx = torch.randint(0, T, [B], device=x.device)
            frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
            frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
            frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

            return frames, frames_recon, x, x_recon
        finally:
            # Restore training mode
            if was_training:
                self.train()

    def log_images(self, batch, **kwargs):
        log = dict()
        batch = batch.to(self.device)
        frames, frames_rec, _, _ = self.inference(batch)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        _, _, batch, x_rec = self.inference(batch)
        log["inputs"] = batch
        log["reconstructions"] = x_rec
        return log


def normalize(in_channels, norm_type='group', num_groups=32):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        num_groups = min(num_groups, in_channels)  # Ensure valid grouping
        if in_channels % num_groups != 0:
            num_groups = 8  # Fallback to 8 groups
        return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)
    else:
        return None


class Encoder(nn.Module):
    def __init__(self, img_type, n_hiddens, image_channel=3, norm_type='group', num_groups=32):
        super().__init__()
        if img_type == 'mri':
            kernels = [5, 3]
            strides = [(5, 2, 5), (1, 3, 1)]
        elif img_type == 'pet':
            kernels = [3, 3]
            strides = [(2, 2, 3), (2, 2, 1)]
        else:
            kernels = [3, 3]
            strides = [2, 2]
        self.conv_blocks = nn.ModuleList()
        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3)

        out_channels = n_hiddens
        in_channels = n_hiddens
        for i in range(len(kernels)):
            out_channels = in_channels * 2
            block = nn.Module()
            block.down = SamePadConv3d(in_channels, out_channels, kernels[i], stride=strides[i])
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            in_channels = out_channels

        self.final_block = nn.Sequential(
            normalize(out_channels, norm_type, num_groups=num_groups),
            nn.SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, type, n_hiddens, image_channel, norm_type='group', num_groups=32):
        super().__init__()
        if type == 'mri':
            kernels = [(1, 3, 1), (5, 2, 5)]
        elif type == 'pet':
            kernels = [(2, 2, 1), (2, 2, 3)]
        else:
            kernels = [2, 2]

        in_channels = n_hiddens * 4
        self.final_block = nn.Sequential(
            normalize(in_channels, norm_type, num_groups=num_groups),
            nn.SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        out_channels = n_hiddens * 4
        for i in range(len(kernels)):
            block = nn.Module()
            out_channels = in_channels // 2
            block.up = nn.ConvTranspose3d(in_channels, out_channels, kernels[i], stride=kernels[i])
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type, num_groups=num_groups)
            self.conv_blocks.append(block)
            in_channels = out_channels

        self.conv_last = SamePadConv3d(out_channels, image_channel, kernel_size=3)

    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group',
                 padding_type='replicate', num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels, norm_type, num_groups=num_groups)
        self.conv1 = SamePadConv3d(
            in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = normalize(out_channels, norm_type, num_groups=num_groups)
        self.conv2 = SamePadConv3d(
            out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(
                in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + h


class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3
        # Calculate padding for 'same' behavior
        padding = tuple((k - 1) // 2 for k in kernel_size)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                              padding_mode=padding_type, bias=bias)

    def forward(self, x):
        return self.conv(x)
