import subprocess
from pprint import pprint

import nibabel as nib
import numpy as np


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


def extract_patches():
    patches = [[] for _ in range(61)]
    seg_mri = nib.load('seg/preprocessed_seg.nii')
    seg_mri = np.array(seg_mri.get_fdata(), dtype=np.int16)
    for i in range(seg_mri.shape[0]):
        for j in range(seg_mri.shape[1]):
            for k in range(seg_mri.shape[2]):
                patches[seg_mri[i, j, k]].append((i, j, k))

    pprint(patches)

# seg_mri = nib.load('icbm_avg_152_t1_tal_nlin_symmetric_VI_synthseg.nii').get_fdata()
