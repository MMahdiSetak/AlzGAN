import torch.nn as nn


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
