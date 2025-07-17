import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchio as tio


class MRIDataset(Dataset):
    def __init__(self, mri, labels, split, apply_augmentation=False):
        # self.data_path = data_path
        self.mri_images, self.labels = mri, labels
        self.split = split
        # self.file = None
        self.apply_augmentation = apply_augmentation and (split == 'train')
        self.augmentation_transform = self._create_augmentation_pipeline()

    def _create_augmentation_pipeline(self):
        """Create TorchIO augmentation pipeline optimized for 3D MRI (160, 192, 160)"""
        base_transforms = [
            # tio.ZNormalization(masking_method=tio.ZNormalization.mean),  # Z-score normalization
            tio.ZNormalization(masking_method=lambda x: x > 0),  # Z-score normalization
            tio.Resize(target_shape=(80, 96, 80)),
        ]
        if not self.apply_augmentation:
            # return None
            return tio.Compose(base_transforms)

        # Single augmentation approach - apply one technique at a time
        augmentation_transforms = [
            tio.RandomFlip(axes='LR', p=0.5),
            # Primary augmentations (choose one per batch)
            tio.OneOf({
                # Elastic deformation (most effective for brain MRI)
                tio.RandomElasticDeformation(
                    num_control_points=7,
                    max_displacement=7.5,
                    locked_borders=2
                ): 0.35,
                # Intensity augmentation (second most effective)
                tio.RandomGamma(log_gamma=0.3): 0.30,
                # Spatial transformations
                tio.RandomAffine(
                    scales=(0.8, 1.2),  # ±20% scaling
                    degrees=15,  # ±15° rotation
                    translation=8  # Conservative translation (5% of 160)
                ): 0.25,
                # No augmentation
                tio.Lambda(lambda x: x): 0.1
            }),
            tio.Resize(target_shape=(80, 96, 80)),
        ]
        return tio.Compose(base_transforms + augmentation_transforms)

    def __len__(self):
        # if self.file is None:
        #     with h5py.File(self.data_path, 'r') as f:
        #         return len(f[f'label_{self.split}'])
        # return len(self.file[f'label_{self.split}'])
        return len(self.labels)

    def get_class_weights(self):
        """
        Calculate class weights based on inverse frequency of labels.
        Returns a dictionary mapping class labels to their weights.
        """
        # with h5py.File(self.data_path, 'r') as f:
        #     labels = f[f'label_{self.split}'][:]
        labels = self.labels[:]

        unique, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        class_weights = {cls: total_samples / (len(unique) * count) for cls, count in zip(unique, counts)}

        return class_weights

    def __getitem__(self, index):
        # if self.file is None:
        #     self.file = h5py.File(self.data_path, 'r')
        #     self.mri_images = self.file[f'mri_{self.split}']
        #     self.labels = self.file[f'label_{self.split}']

        # mri_tensor = torch.from_numpy(self.mri_images[index].astype(np.float32)).unsqueeze(0)
        # label_tensor = torch.tensor(self.labels[index], dtype=torch.long)

        mri_tensor = self.mri_images[index].type(torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)

        # Apply augmentation if enabled
        if self.augmentation_transform is not None:
            # Create TorchIO Subject
            subject = tio.Subject(
                mri=tio.ScalarImage(tensor=mri_tensor),
                label=label_tensor
            )
            # Apply transforms
            try:
                augmented_subject = self.augmentation_transform(subject)
                mri_tensor = augmented_subject['mri'].data
                del augmented_subject, subject
            except Exception as e:
                print(f"Warning: Augmentation failed for sample {index}: {e}")
                pass

        return mri_tensor, label_tensor


class SimpleMRIDataset(Dataset):
    def __init__(self, data_path, split, apply_augmentation=False):
        self.data_path = data_path
        self.split = split
        self.file = None

    def __len__(self):
        if self.file is None:
            with h5py.File(self.data_path, 'r') as f:
                return len(f[f'label_{self.split}'])
        return len(self.file[f'label_{self.split}'])

    def get_class_weights(self):
        """
        Calculate class weights based on inverse frequency of labels.
        Returns a dictionary mapping class labels to their weights.
        """
        with h5py.File(self.data_path, 'r') as f:
            labels = f[f'label_{self.split}'][:]

        unique, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        class_weights = {cls: total_samples / (len(unique) * count) for cls, count in zip(unique, counts)}

        return class_weights

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')
            self.mri_images = self.file[f'mri_{self.split}']
            self.labels = self.file[f'label_{self.split}']

        mri_tensor = torch.from_numpy(self.mri_images[index].astype(np.float32)).unsqueeze(0)
        label_tensor = torch.tensor(self.labels[index], dtype=torch.long)

        return {'mri': mri_tensor, 'label': label_tensor}
        # return self.mri_images[index], self.labels[index]


class MRIRAMLoader:
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.mri_images = None
        self.labels = None

    def get_data(self):
        if self.mri_images is None:
            with h5py.File(self.data_path, 'r') as f:
                self.mri_images = torch.from_numpy(f[f'mri_{self.split}'][:]).share_memory_()
                labels_tensor = torch.from_numpy(f[f'label_{self.split}'][:]).long()
                _, mapped_labels = torch.unique(labels_tensor, sorted=True, return_inverse=True)
                self.labels = mapped_labels.share_memory_()
        return self.mri_images, self.labels


class FastMRIDataset(Dataset):
    def __init__(self, mri, labels, transform=False):
        self.mri_images, self.labels = mri, labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def get_class_weights(self):
        """
        Calculate class weights based on inverse frequency of labels.
        Returns a dictionary mapping class labels to their weights.
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        class_weights = {cls: total_samples / (len(unique) * count) for cls, count in zip(unique, counts)}

        return class_weights

    def __getitem__(self, index):
        mri = self.mri_images[index]
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
                self.labels = torch.from_numpy(f[f'label_{self.split}'][:]).long().share_memory_()
        return self.pet_images, self.labels


class FastPETDataset(Dataset):
    def __init__(self, pet, labels):
        self.pet_images, self.labels = pet, labels
        self.label_mapping = torch.tensor([0, 0, 1, 1, 1, 2])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        mri = self.pet_images[index]
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
        pet = torch.from_numpy(self.pet_images[index].astype(np.float32)).div_(127.5).sub_(1).unsqueeze(0)
        # pet = F.interpolate(pet, size=(128, 128, 128), mode='trilinear', align_corners=False)
        label = int(self.labels[index])
        return mri.squeeze(0), pet, label


class MRI2PETDataset(Dataset):
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
        mri = F.interpolate(mri, size=(64, 64, 64), mode='trilinear', align_corners=False)
        pet = torch.from_numpy(self.pet_images[index].astype(np.float32)).div_(127.5).sub_(1).unsqueeze(0).unsqueeze(0)
        pet = F.interpolate(pet, size=(64, 64, 64), mode='trilinear', align_corners=False)
        label = int(self.labels[index])
        return mri.squeeze(0), pet.squeeze(0), label


class VQGANDataset(Dataset):
    def __init__(self, data_path, split, modality='mri'):
        self.data_path = data_path
        self.split = split
        self.file = None
        self.modality = modality

    def __len__(self):
        if self.file is None:
            with h5py.File(self.data_path, 'r') as f:
                return len(f[f'label_{self.split}'])
        return len(self.file[f'label_{self.split}'])

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.data_path, 'r')
            self.images = self.file[f'{self.modality}_{self.split}']
        img = torch.from_numpy(self.images[index].astype(np.float32)).div_(127.5).sub_(1).unsqueeze_(0).unsqueeze(0)
        img = F.interpolate(img, size=(64, 64, 64), mode='trilinear', align_corners=False)
        return img.squeeze(0)


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
        mri = self.mri_images[index]
        pet = self.pet_images[index]
        label = int(self.labels[index])
        return mri, pet, label
