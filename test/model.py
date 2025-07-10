from torchsummary import summary
from torch import nn


class MRICNN(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3, channels=64):
        super(MRICNN, self).__init__()

        self.model = nn.Sequential(
            # CNN blocks
            # nn.Conv3d(1, 1, kernel_size=3, padding=1),
            # nn.BatchNorm3d(1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=(2, 2, 3), stride=(2, 2, 3)),
            # nn.Dropout3d(0.2),

            # nn.Conv3d(1, 4, kernel_size=3, padding=1),
            # nn.BatchNorm3d(4),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # nn.Dropout3d(0.1),

            nn.Conv3d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            # nn.Dropout3d(0.1),

            nn.Conv3d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
            # nn.Dropout3d(0.1),

            nn.Conv3d(channels * 2, channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=4, stride=4),
            #
            # nn.Conv3d(channels * 4, channels * 8, kernel_size=3, padding=1),
            # nn.BatchNorm3d(channels * 8),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=2, stride=2),

            # nn.Conv3d(channels * 8, channels * 16, kernel_size=3, padding=1),
            # nn.BatchNorm3d(channels * 16),
            # nn.ReLU(inplace=True),
            # nn.MaxPool3d(kernel_size=2, stride=2),

            # Global average pooling and flatten
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),

            # Fully connected layers
            nn.Dropout(dropout_rate),
            nn.Linear(channels * 4, channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            # nn.Linear(channels * 8, channels * 4),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout_rate),
            nn.Linear(channels * 4, num_classes),
        )

    def forward(self, x):
        return self.model(x)


classifier = MRICNN(num_classes=3, dropout_rate=0.6, channels=32).to('cuda')

summary(classifier, (1, 160, 192, 160))
