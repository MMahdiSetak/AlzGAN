import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


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

        # mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(256)
        mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(127.5).sub_(1).unsqueeze(0).unsqueeze(0)
        mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        label = int(self.labels[index])
        return mri.squeeze(0), label


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


class DDPMPairDataset(Dataset):
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
        mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(127.5).sub_(1).unsqueeze(0).unsqueeze(0)
        mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        pet = torch.from_numpy(self.pet_images[index].astype(np.float32)).div_(127.5).sub_(1).unsqueeze(0).unsqueeze(0)
        pet = F.interpolate(pet, size=(128, 128, 128),
                            mode='trilinear', align_corners=False)
        label = int(self.labels[index])
        return mri.squeeze(0), pet.squeeze(0), label


class TempPairDataset(Dataset):
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
        mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(255).unsqueeze(0).unsqueeze(0)
        mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        pet = torch.from_numpy(self.pet_images[index].astype(np.float32)).div_(255).unsqueeze(0).unsqueeze(0)
        pet = F.interpolate(pet, size=(128, 128, 128),
                            mode='trilinear', align_corners=False)
        label = int(self.labels[index])
        return mri.squeeze(0), pet.squeeze(0), label
