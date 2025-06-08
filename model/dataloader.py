import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import monai.transforms as T


class MRIDataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.file = None
        if split == 'train':
            self.train_transforms = T.Compose([
                T.RandRotate(range_x=np.pi / 18, range_y=np.pi / 18, range_z=np.pi / 18, prob=0.5),  # ±10 degrees
                T.Rand3DElastic(
                    sigma_range=(2, 5), magnitude_range=(0.1, 0.3), prob=0.3
                ),
                T.RandAffine(
                    translate_range=(10, 10, 10), scale_range=(-0.1, 0.1), prob=0.5
                ),
                T.RandGaussianNoise(std=0.01, prob=0.2),  # Light noise
                T.RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),  # Gamma correction
                T.RandBiasField(prob=0.3)
            ])

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

        # mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(255)
        if self.split == 'train':
            mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(255).unsqueeze(0)
            mri = self.train_transforms(mri)
            mri = mri.multiply_(2).sub_(1).unsqueeze(0)
            mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        else:
            mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(127.5).sub_(1).unsqueeze(
                0).unsqueeze(0)
            mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
        label = int(self.labels[index])
        return mri.squeeze(0), label


class MRIRAMLoader:
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.mri_images = None
        self.labels = None

    def get_data(self):
        if self.mri_images is None:
            with h5py.File(self.data_path, 'r') as f:
                self.mri_images = torch.from_numpy(f[f'mri_{self.split}'][:].astype(np.float32)).share_memory_()
                self.labels = torch.from_numpy(f[f'label_{self.split}'][:]).long().share_memory_()
        return self.mri_images, self.labels


class FastMRIDataset(Dataset):
    def __init__(self, mri, labels, transform=False):
        self.mri_images, self.labels = mri, labels
        self.transform = transform
        self.train_transforms = T.Compose([
            T.RandRotate(range_x=np.pi / 18, range_y=np.pi / 18, range_z=np.pi / 18, prob=0.3),
            T.Rand3DElastic(sigma_range=(2, 5), magnitude_range=(0.1, 0.3), prob=0.3),
            T.RandAffine(translate_range=(10, 10, 10), scale_range=(-0.1, 0.1), prob=0.5),
            T.RandGaussianNoise(std=0.01, prob=0.2),
            T.RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),
            T.RandBiasField(prob=0.3)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mri = self.mri_images[index]
        if self.transform:
            mri = self.train_transforms(mri.div_(255).unsqueeze(0))
        label = self.labels[index]
        return mri, label


class PETRAMLoader:
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.pet_images = None
        self.labels = None

    def get_data(self):
        if self.pet_images is None:
            with h5py.File(self.data_path, 'r') as f:
                self.pet_images = torch.from_numpy(f[f'pet_{self.split}'][:].astype(np.float32)).share_memory_()
                self.labels = torch.from_numpy(f[f'label_{self.split}'][:]).share_memory_()
        return self.pet_images, self.labels


class FastPETDataset(Dataset):
    def __init__(self, pet, labels):
        self.pet_images, self.labels = pet, labels
        self.label_mapping = torch.tensor([0, 0, 1, 1, 1, 2])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mri = self.pet_images[index].div(127.5).sub_(1).unsqueeze_(0)
        label = self.labels[index]
        return mri, self.label_mapping[label]


# class FastMRIDataset(Dataset):
#     def __init__(self, data_path, split):
#         self.data_path = data_path
#         self.split = split
#         with h5py.File(self.data_path, 'r') as f:
#             self.mri_images = f[f'mri_{split}'][:]  # load into RAM
#             self.labels = f[f'label_{split}'][:]
#         if split == 'train':
#             self.train_transforms = T.Compose(
#                 [
#                     T.RandRotate(range_x=np.pi / 18, range_y=np.pi / 18, range_z=np.pi / 18, prob=0.5),  # ±10 degrees
#                     T.Rand3DElastic(
#                         sigma_range=(2, 5), magnitude_range=(0.1, 0.3), prob=0.3
#                     ),
#                     T.RandAffine(
#                         translate_range=(10, 10, 10), scale_range=(-0.1, 0.1), prob=0.5
#                     ),
#                     T.RandGaussianNoise(std=0.01, prob=0.2),  # Light noise
#                     T.RandAdjustContrast(gamma=(0.8, 1.2), prob=0.3),  # Gamma correction
#                     T.RandBiasField(prob=0.3)
#                 ],
#                 overrides={'device': 'cuda'}
#             )
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         if self.split == 'train':
#             mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(255).unsqueeze(0)
#             mri = self.train_transforms(mri)
#             mri = mri.multiply_(2).sub_(1).unsqueeze(0)
#             mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
#         else:
#             mri = torch.from_numpy(self.mri_images[index].astype(np.float32)).div_(127.5).sub_(1).unsqueeze(
#                 0).unsqueeze(0)
#             mri = F.interpolate(mri, size=(128, 128, 128), mode='trilinear', align_corners=False)
#         label = int(self.labels[index])
#         return mri.squeeze(0), label


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
