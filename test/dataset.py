import matplotlib.pyplot as plt

from model.dataloader import MRIDataset

train_ds = MRIDataset('../dataset/mri_label_v3.hdf5', 'train')
for i in range(10):
    # mri, pet, label = next(iter(train_ds))
    mri, label = train_ds[26 * (i + 1)]
    plt.imshow(mri[0, :, :, 64], cmap='gray')
    plt.title('Augmented MRI')
    plt.axis('off')
    plt.show()
