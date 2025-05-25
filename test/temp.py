from torch.utils.data import DataLoader

from model.dataloader import DDPMPairDataset, PairDataset, TempPairDataset
from torchmetrics.image import PeakSignalNoiseRatio

# train_loader = DataLoader(
#     dataset=DDPMPairDataset('../dataset/mri_pet_label_v3.hdf5', 'train'),
#     batch_size=1, num_workers=1, shuffle=False, drop_last=False
# )
# val_loader = DataLoader(
#     dataset=DDPMPairDataset('../dataset/mri_pet_label_v3.hdf5', 'val'),
#     batch_size=1, num_workers=1, shuffle=False, drop_last=False
# )
# train_ds = DDPMPairDataset('../dataset/mri_pet_label_v3.hdf5', 'train')
# train_ds = PairDataset('../dataset/mri_pet_label_v3.hdf5', 'train')
train_ds = TempPairDataset('../dataset/mri_pet_label_v3.hdf5', 'train')
psnr = PeakSignalNoiseRatio(data_range=1)
for i in range(10):
    # mri, pet, label = next(iter(train_ds))
    # mri, pet, label = train_ds[26 * (i + 1)]
    b_mri, b_pet, b_label = train_ds[126 * (i + 1)]
    print(psnr(b_pet, b_mri))
