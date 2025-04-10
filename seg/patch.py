import subprocess
from pprint import pprint

import nibabel as nib
import numpy as np

# labels = [
#     "0", "2", "3", "4", "5", "7", "8", "10", "11", "12", "13", "14", "15", "16", "17", "18", "24", "26", "28",
#     "41", "42", "43", "44", "46", "47", "49", "50", "51", "52", "53", "54", "58", "60"
# ]
labels = [
    0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26, 28, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52,
    53, 54, 58, 60
]


def template_preprocess():
    command = f'docker run --rm --gpus all -v ./seg:/temp freesurfer/synthstrip:1.6 -i /temp/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii -o /temp/stripped.nii'
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    if proc.returncode != 0:
        print(f'stripping failed: {proc.returncode}')

    mri_template = nib.load('seg/stripped.nii')
    img = mri_template.get_fdata()
    cropped_img = img[15:175, 17:217, :180]
    new_nifti = nib.Nifti1Image(cropped_img, mri_template.affine)
    nib.save(new_nifti, "seg/preprocessed_template.nii")


def segmentation_preprocess():
    seg_template = nib.load('seg/icbm_avg_152_t1_tal_nlin_symmetric_VI_synthseg.nii')
    img = seg_template.get_fdata()
    cropped_img = img[15:175, 17:217, :180]
    new_nifti = nib.Nifti1Image(cropped_img, seg_template.affine)
    nib.save(new_nifti, "seg/preprocessed_seg.nii")


def get_patch_indices():
    seg_mri = nib.load('seg/preprocessed_seg.nii')
    seg_mri = np.array(seg_mri.get_fdata(), dtype=np.int16)
    indices = {}
    for label in labels:
        indices[label] = seg_mri == label
    return indices


def extract_patches():
    patches = [[] for _ in range(61)]
    seg_mri = nib.load('seg/preprocessed_seg.nii')
    seg_mri = np.array(seg_mri.get_fdata(), dtype=np.int16)
    for i in range(seg_mri.shape[0]):
        for j in range(seg_mri.shape[1]):
            for k in range(seg_mri.shape[2]):
                patches[seg_mri[i, j, k]].append((i, j, k))
    return patches
    # pprint(patches)


def print_patch_sizes():
    patches = extract_patches()
    print("0:", len(patches[0]))
    print("2:", len(patches[2]))
    print("3:", len(patches[3]))
    print("4:", len(patches[4]))
    print("5:", len(patches[5]))
    print("7:", len(patches[7]))
    print("8:", len(patches[8]))
    print("10:", len(patches[10]))
    print("11:", len(patches[11]))
    print("12:", len(patches[12]))
    print("13:", len(patches[13]))
    print("14:", len(patches[14]))
    print("15:", len(patches[15]))
    print("16:", len(patches[16]))
    print("17:", len(patches[17]))
    print("18:", len(patches[18]))
    print("24:", len(patches[24]))
    print("26:", len(patches[26]))
    print("28:", len(patches[28]))
    print("41:", len(patches[41]))
    print("42:", len(patches[42]))
    print("43:", len(patches[43]))
    print("44:", len(patches[44]))
    print("46:", len(patches[46]))
    print("47:", len(patches[47]))
    print("49:", len(patches[49]))
    print("50:", len(patches[50]))
    print("51:", len(patches[51]))
    print("52:", len(patches[52]))
    print("53:", len(patches[53]))
    print("54:", len(patches[54]))
    print("58:", len(patches[58]))
    print("60:", len(patches[60]))

# seg_mri = nib.load('icbm_avg_152_t1_tal_nlin_symmetric_VI_synthseg.nii').get_fdata()
