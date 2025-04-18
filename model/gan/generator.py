import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 base_nf: int = 128,
                 enc_dropout: float = 0.1,
                 bottle_dropout: float = 0.2,
                 dec_dropout: float = 0.1,
                 n_bottleneck: int = 3):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, base_nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_nf),
            nn.ReLU(inplace=True), nn.Dropout3d(enc_dropout),

            nn.Conv3d(base_nf, base_nf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_nf * 2),
            nn.ReLU(inplace=True), nn.Dropout3d(enc_dropout),

            nn.Conv3d(base_nf * 2, base_nf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_nf * 4),
            nn.ReLU(inplace=True), nn.Dropout3d(enc_dropout),

            nn.Conv3d(base_nf * 4, base_nf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_nf * 8),
            nn.ReLU(inplace=True), nn.Dropout3d(enc_dropout),
        )

        bottle_blocks = []
        for _ in range(n_bottleneck):
            bottle_blocks += [
                nn.Conv3d(base_nf * 8, base_nf * 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(base_nf * 8),
                nn.LeakyReLU(0.2),
                nn.Dropout3d(bottle_dropout),
            ]
        self.bottleneck = nn.Sequential(*bottle_blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_nf * 8, base_nf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_nf * 4),
            nn.ReLU(inplace=True), nn.Dropout3d(dec_dropout),

            nn.ConvTranspose3d(base_nf * 4, base_nf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_nf * 2),
            nn.ReLU(inplace=True), nn.Dropout3d(dec_dropout),

            nn.ConvTranspose3d(base_nf * 2, base_nf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(base_nf),
            nn.ReLU(inplace=True), nn.Dropout3d(dec_dropout),

            nn.ConvTranspose3d(base_nf, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        return x[:, :, :100, :140, :96]
