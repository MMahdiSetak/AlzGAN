import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.file = None

    def __len__(self):
        if self.file is None:
            with h5py.File(self.data_path, 'r') as f:
                return len(f[f'label_{self.split}'])
        return len(self.file[f'label_{self.split}'])

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')
            self.mri_images = self.file[f'mri_{self.split}']
            self.labels = self.file[f'label_{self.split}']

        mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(256)
        label = int(self.labels[index])
        return mri, label


class PairDataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.file = None

    def __len__(self):
        if self.file is None:
            with h5py.File(self.data_path, 'r') as f:
                return len(f[f'label_{self.split}'])
        return len(self.file[f'label_{self.split}'])

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')
            self.mri_images = self.file[f'mri_{self.split}']
            self.pet_images = self.file[f'pet_{self.split}']
            self.labels = self.file[f'label_{self.split}']
        mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(256).unsqueeze(0)
        pet = torch.from_numpy(self.pet_images[index].astype(np.float32)).div_(256).unsqueeze(0)
        label = int(self.labels[index])
        return mri, pet, label
