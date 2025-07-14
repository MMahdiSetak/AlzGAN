import ants
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# from model.dataloader import MRIDataset

def log_to_file_image(img, file_name='test'):
    center_slices = [dim // 2 for dim in img.shape]
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    # titles = ['Axial', 'Coronal', 'Sagittal']
    slices = [img[center_slices[0], :, :], img[:, center_slices[1], :], img[:, :, center_slices[2]]]
    for ax, slice_img in zip(axes, slices):
        ax.imshow(slice_img, cmap='gray')
        # ax.set_title(title)
        # ax.axis('off')
    plt.show()
    plt.close(fig)


# org_img = nib.load(
#     "../dataset/MRI2/ADNI/003_S_0981/FreeSurfer_Longitudinal_Processing_brainmask/2007-05-02_08_18_48.0/I209483/brainmask.mgz").get_fdata().astype(
#     np.uint8)


# mri_template = ants.image_read('../template/mni_icbm152_nl_VI_nifti/stripped_cropped.nii')
# moving_image = ants.image_read(
#     "../dataset/MRI2/ADNI/003_S_0981/FreeSurfer_Longitudinal_Processing_brainmask/2007-05-02_08_18_48.0/I209483/brainmask.mgz",
#     pixeltype='unsigned char')
# registration = ants.registration(fixed=mri_template, moving=moving_image, type_of_transform='Rigid')
# org_img = registration["warpedmovout"].numpy()

orig = nib.load("../template/mni_icbm152_nl_VI_nifti/stripped.nii")
img = orig.get_fdata()
img_cropped = img[16:176, 20:212, 5:165]
# img_cropped = np.transpose(img_cropped, (0, 2, 1))
# img_cropped = img_cropped[:, ::-1, :]
# log_to_file_image(org_img)
log_to_file_image(img_cropped)

# T_crop = np.eye(4)
# T_crop[:3, 3] = (13, 20, 5)

# 3b. Permutation for transpose (new i→old i mapping)
#    new_i = (0→0, 1→2, 2→1)
# P = np.eye(4)
# P[1, 1] = 0
# P[1, 2] = 1
# P[2, 2] = 0
# P[2, 1] = 1

# 3c. Flip along axis 1 of the transposed data
# dim1 = img_cropped.shape[1]
# F = np.eye(4)
# F[1, 1] = -1
# F[1, 3] = dim1 - 1

# Combine them: for a new-voxel index v_new,
#   v_old = T_crop @ P @ F @ [v_new; 1]
# voxel_transform = T_crop @ P @ F

# --- 4. Update affine ---
# new_affine = orig.affine @ voxel_transform

# --- 5. Save ---
# out_nii = nib.Nifti1Image(img_cropped, new_affine, header=orig.header)
out_nii = nib.Nifti1Image(img_cropped, orig.affine, header=orig.header)
nib.save(out_nii, "../template/mni_icbm152_nl_VI_nifti/stripped_cropped.nii")

# train_ds = MRIDataset('../dataset/mri_label_v3.hdf5', 'train')
# for i in range(10):
#     # mri, pet, label = next(iter(train_ds))
#     mri, label = train_ds[26 * (i + 1)]
#     plt.imshow(mri[0, :, :, 64], cmap='gray')
#     plt.title('Augmented MRI')
#     plt.axis('off')
#     plt.show()
