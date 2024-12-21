import io
import os
import sys
from datetime import datetime

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
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
        if split == 'train':
            mri_images = file['mri_train']
            labels = file['label_train']
        elif split == 'val':
            mri_images = file['mri_val']
            labels = file['label_val']
        else:
            mri_images = file['mri_test']
            labels = file['label_test']
        while True:
            for i in range(0, len(mri_images), batch_size):
                end = min(i + batch_size, len(mri_images))
                batch_mri = mri_images[i:end]
                batch_label = labels[i:end]

                if batch_mri.shape[0] != batch_size:  # TODO improve this
                    continue

                batch_mri = torch.Tensor(batch_mri / 256).unsqueeze(1).to(device)  # Add channel dimension
                yield batch_mri, batch_label


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
        enc = self.encoder(x)
        bot = self.bottleneck(enc)
        dec = self.decoder(bot)
        # return dec[:, :, :35, :64, :64]  # Cropping to match output shape
        return enc, bot


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.3),

            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.3),

            nn.AdaptiveAvgPool3d((1, 1, 1)),  # Ensure 3D pooling
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.model(x)


# Initialize models
generator = Generator().to(device)
# generator.load_state_dict(torch.load('model/GAN/best_generator.pth', weights_only=True))
classifier = Classifier().to(device)

# Optimizers
optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_c = optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
classification_loss = nn.CrossEntropyLoss()

# TensorBoard writer
time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
log_dir = os.path.join("log/gan_class", time_stamp)
writer = SummaryWriter(log_dir=log_dir)


def train_gan_class(generator, classifier, data_generator, val_data_generator, epochs=100, steps_per_epoch=100,
                    steps_per_epoch_val=50):
    model_path = f"model/GAN_class/{time_stamp}"
    os.makedirs(model_path, exist_ok=True)
    best_acc = 0

    for epoch in range(epochs):
        generator.train()
        classifier.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}", unit="step", leave=False) as pbar:
            for step in range(steps_per_epoch):
                # Get data batch
                mri, labels = next(data_generator)
                labels = torch.tensor(labels).to(device)

                optimizer_c.zero_grad()
                optimizer_g.zero_grad()

                enc_features, bottleneck_features = generator(mri)
                stacked_features = torch.cat((enc_features, bottleneck_features), dim=1)
                classifier_output = classifier(stacked_features)

                # Classification loss
                cls_loss = classification_loss(classifier_output, labels)

                cls_loss.backward()
                optimizer_c.step()
                optimizer_g.step()

                # Calculate accuracy
                train_correct = (classifier_output.argmax(dim=1) == labels).sum().item()
                train_total = labels.size(0)
                train_accuracy = 100 * train_correct / train_total

                # Accumulate for epoch-level metrics
                epoch_train_loss += cls_loss.item()
                epoch_train_correct += train_correct
                epoch_train_total += train_total

                # Update progress bar
                pbar.set_postfix({"Train Loss": f"{cls_loss.item():.4f}", "Train Acc": f"{train_accuracy:.2f}"})
                pbar.update(1)

        # Validation Phase
        generator.eval()
        classifier.eval()

        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        with torch.no_grad():
            for step in range(steps_per_epoch_val):
                mri, labels = next(val_data_generator)
                mri = mri.to(device)
                labels = torch.tensor(labels).to(device)

                # Forward pass through generator and classifier
                enc_features, bottleneck_features = generator(mri)
                stacked_features = torch.cat((enc_features, bottleneck_features), dim=1)

                # Classifier validation loss
                classifier_output = classifier(stacked_features)
                val_loss = classification_loss(classifier_output, labels).item()

                # Calculate accuracy
                val_correct = (classifier_output.argmax(dim=1) == labels).sum().item()
                val_total = labels.size(0)

                epoch_val_loss += val_loss
                epoch_val_correct += val_correct
                epoch_val_total += val_total

            # Epoch-level metrics
            epoch_train_loss /= steps_per_epoch
            epoch_train_accuracy = 100 * epoch_train_correct / epoch_train_total

            epoch_val_loss /= steps_per_epoch_val
            epoch_val_accuracy = 100 * epoch_val_correct / epoch_val_total

            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.2f}, "
                f"Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.2f}"
            )
            writer.add_scalar("Loss/train", epoch_train_loss, epoch + 1)
            writer.add_scalar("Loss/val", epoch_val_loss, epoch + 1)
            writer.add_scalar("Acc/train", epoch_train_accuracy, epoch + 1)
            writer.add_scalar("Acc/val", epoch_val_accuracy, epoch + 1)

            # Save model
            if epoch_val_accuracy > best_acc:
                best_acc = epoch_val_accuracy
                torch.save(generator.state_dict(), os.path.join(model_path, f"generator_epoch_{epoch + 1}.pth"))
                torch.save(classifier.state_dict(), os.path.join(model_path, "classifier.pth"))
                print("Classifier model saved")


def save_model_metadata(model, input, name, batch_size):
    with open(f"{log_dir}/{name}.txt", "w") as f:
        buffer = io.StringIO()
        sys.stdout = buffer
        summary(model, input, batch_size=batch_size)
        sys.stdout = sys.__stdout__
        out = buffer.getvalue()
        f.write(out)


# Example usage
hdf5_file = 'dataset/mri_label_small.hdf5'
with h5py.File(hdf5_file, 'r') as file:
    train_size = len(file['label_train'])
    val_size = len(file['label_val'])
    test_size = len(file['label_test'])
batch_size = 12
# train_size = 2124
# train_size = 983
# val_size = 266

save_model_metadata(generator, mri_target_shape, "generator", batch_size)
save_model_metadata(classifier, (2 * 256, 10, 12, 12), "classifier", batch_size)

steps_per_epoch = train_size // batch_size
steps_per_epoch_val = val_size // batch_size
train_data_generator = pair_data_generator(hdf5_file, batch_size, "train")
val_data_generator = pair_data_generator(hdf5_file, batch_size, "val")
train_gan_class(generator, classifier, train_data_generator, val_data_generator, epochs=100,
                steps_per_epoch=steps_per_epoch, steps_per_epoch_val=steps_per_epoch_val)
