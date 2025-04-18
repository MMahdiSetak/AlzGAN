import h5py
import torch
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.file = h5py.File(self.data_path, 'r')
        self.mri_images = self.file[f'mri_{split}']
        self.labels = self.file[f'label_{split}']
        self.n = len(self.mri_images)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        mri = torch.Tensor(self.mri_images[index] / 256)
        label = self.labels[index]
        return mri, label


class PairDataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.file = h5py.File(self.data_path, 'r')
        self.mri_images = self.file[f'mri_{split}']
        self.pet_images = self.file[f'pet_{split}']
        self.labels = self.file[f'label_{split}']
        self.n = len(self.mri_images)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        mri = torch.Tensor(self.mri_images[index] / 256).unsqueeze(1)
        pet = torch.Tensor(self.pet_images[index] / 256).unsqueeze(1)
        label = self.labels[index]
        return mri, pet, label
