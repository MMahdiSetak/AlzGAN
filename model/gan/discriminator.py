import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 base_nf: int = 64,
                 n_layers: int = 5,
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 use_bn: bool = True,
                 dropout: float = 0.1, ):
        super(Discriminator, self).__init__()
        layers = []
        # first conv: no down‐sampling
        layers += [
            nn.Conv3d(1, base_nf, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(dropout),
        ]

        # intermediate blocks
        for i in range(1, n_layers):
            in_ch = base_nf * (2 ** (i - 1))
            out_ch = base_nf * (2 ** i)
            layers += [
                nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            ]
            if use_bn:
                layers.append(nn.BatchNorm3d(out_ch))
            layers += [
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout3d(dropout),
            ]

        # final conv → single‐channel “real/fake” score
        last_ch = base_nf * (2 ** (n_layers - 1))
        layers += [
            nn.Conv3d(last_ch, 1, kernel_size=kernel_size, stride=1, padding=0),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
