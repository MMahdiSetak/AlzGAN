import math

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# device = 'cpu'


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(DataLoader):
    def __init__(self, data_path, split, pet=False, label=True):
        self.data_path = data_path
        self.split = split
        self.pet = pet
        self.label = label

        # Open the h5 file to read the data
        with h5py.File(self.data_path, 'r') as file:
            mri_images = file[f'mri_{split}']
            if pet:
                self.pet_images = file[f'pet_{split}']
            if label:
                self.labels = file[f'label_{split}']
            self.n = len(mri_images)

        self.indices = np.arange(self.n)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        # Get the index and return the data
        # batch_indices = self.indices[index]
        with h5py.File(self.data_path, 'r') as file:
            # MRI image
            mri_images = file[f'mri_{self.split}']
            batch_mri = torch.Tensor(mri_images[index] / 256)
            # print(f"mri: {batch_indices}, {batch_mri.shape}")

            # Optionally include PET image
            # batch_pet = None
            # if self.pet:
            #     batch_pet = torch.Tensor(self.pet_images[batch_indices] / 256).unsqueeze(0)

            # Optionally include label
            # batch_label = None
            # if self.label:
            labels = file[f'label_{self.split}']
            batch_label = labels[index]
            # print(f"label: {batch_indices}, {batch_label.shape}")

        # return {'mri': batch_mri, 'label': batch_label}
        return batch_mri, batch_label


class DataLoader:
    def __init__(self, data_path, batch_size, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        with h5py.File(self.data_path, 'r') as file:
            self.train_size = len(file['label_train'])
            self.val_size = len(file['label_val'])
            self.test_size = len(file['label_test'])

        self.steps_per_epoch_train = math.ceil(self.train_size / batch_size)
        self.steps_per_epoch_val = math.ceil(self.val_size / batch_size)
        self.steps_per_epoch_test = math.ceil(self.test_size / batch_size)

    def data_loader(self, split, pet=False, label=True):
        """Returns a PyTorch DataLoader for the given split."""
        dataset = CustomDataset(self.data_path, split, pet, label)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,  # Enable shuffling for training
            drop_last=False  # Drop the last incomplete batch if it has fewer than batch_size samples
        )
        return loader

    def get_steps_per_epoch(self, split):
        """Returns the number of steps per epoch for a given split."""
        if split == 'train':
            return self.steps_per_epoch_train
        elif split == 'val':
            return self.steps_per_epoch_val
        elif split == 'test':
            return self.steps_per_epoch_test
        else:
            raise ValueError(f"Unknown split: {split}")

#
# class DataLoader:
#     def __init__(self, data_path, batch_size):
#         self.data_path = data_path
#         self.batch_size = batch_size
#
#         with h5py.File(self.data_path, 'r') as file:
#             self.train_size = len(file['label_train'])
#             self.val_size = len(file['label_val'])
#             self.test_size = len(file['label_test'])
#         self.steps_per_epoch_train = math.ceil(self.train_size / batch_size)
#         self.steps_per_epoch_val = math.ceil(self.val_size / batch_size)
#         self.steps_per_epoch_test = math.ceil(self.test_size / batch_size)
#
#     def data_generator(self, batch_size, split, pet, label):
#         split_dict = {
#             'train': ('mri_train', 'pet_train', 'label_train'),
#             'val': ('mri_val', 'pet_val', 'label_val'),
#             'test': ('mri_test', 'pet_test', 'label_test')
#         }
#         with h5py.File(self.data_path, 'r') as file:
#             mri_images = file[split_dict[split][0]]
#             if pet:
#                 pet_images = file[split_dict[split][1]]
#             if label:
#                 labels = file[split_dict[split][2]]
#
#             n = len(mri_images)
#             indices = np.arange(n)
#             while True:
#                 for i in range(0, n, batch_size):
#                     end = min(i + batch_size, n)
#                     batch_indices = indices[i:end]  # Get batch indices
#
#                     if len(batch_indices) < batch_size:
#                         # Randomly sample additional indices to pad the batch
#                         r = np.random.randint(0, n - batch_size)
#                         additional_indices = indices[r:r + (batch_size - len(batch_indices))]
#                         batch_indices = np.concatenate((additional_indices, batch_indices))
#
#                     batch_output = []
#                     batch_mri = mri_images[batch_indices]
#                     # batch_mri = torch.Tensor(batch_mri / 256).unsqueeze(1).to(device)
#                     batch_mri = torch.Tensor(batch_mri / 256).to(device)
#                     batch_output.append(batch_mri)
#                     if pet:
#                         batch_pet = pet_images[batch_indices]
#                         batch_pet = torch.Tensor(batch_pet / 256).unsqueeze(1).to(device)
#                         batch_output.append(batch_pet)
#                     if label:
#                         batch_label = labels[batch_indices]
#                         batch_output.append(batch_label)
#
#                     yield tuple(batch_output)
#
#     def pet_generator(self, batch_size, split):
#         split_dict = {
#             'train': 'pet_train',
#             'val': 'pet_val',
#             'test': 'pet_test'
#         }
#         with h5py.File(self.data_path, 'r') as file:
#             pet_images = file[split_dict[split]]
#             n = len(pet_images)
#             indices = np.arange(n)
#             while True:
#                 for i in range(0, n, batch_size):
#                     end = min(i + batch_size, n)
#                     batch_indices = indices[i:end]  # Get batch indices
#
#                     if len(batch_indices) < batch_size:
#                         # Randomly sample additional indices to pad the batch
#                         r = np.random.randint(0, n - batch_size)
#                         additional_indices = indices[r:r + (batch_size - len(batch_indices))]
#                         batch_indices = np.concatenate((additional_indices, batch_indices))
#
#                     batch_pet = pet_images[batch_indices]
#                     batch_pet = torch.Tensor(batch_pet / 256).unsqueeze(1).to(device)
#                     yield batch_pet
#
#     def mri_generator(self, batch_size, split):
#         split_dict = {
#             'train': 'mri_train',
#             'val': 'mri_val',
#             'test': 'mri_test'
#         }
#         with h5py.File(self.data_path, 'r') as file:
#             mri_images = file[split_dict[split]]
#             n = len(mri_images)
#             indices = np.arange(n)
#             while True:
#                 for i in range(0, n, batch_size):
#                     end = min(i + batch_size, n)
#                     batch_indices = indices[i:end]  # Get batch indices
#
#                     if len(batch_indices) < batch_size:
#                         # Randomly sample additional indices to pad the batch
#                         r = np.random.randint(0, n - batch_size)
#                         additional_indices = indices[r:r + (batch_size - len(batch_indices))]
#                         batch_indices = np.concatenate((additional_indices, batch_indices))
#
#                     batch_pet = mri_images[batch_indices]
#                     batch_pet = torch.Tensor(batch_pet / 256).unsqueeze(1).to(device)
#                     yield batch_pet
