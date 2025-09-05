import numpy as np
import pandas as pd
import nibabel as nib

from data.image import read_image, skull_stripping
from data.image_tabular import mri_registration
from model.log import log_3d, log_video


def mri_processed():
    df = pd.read_csv("dataset/mri.csv")
    print("Number of images: ", df.shape[0])
    print("Number of subjets: ", len(df['Subject'].unique()))
    img = nib.load(
        "dataset/MRI2/ADNI/005_S_0610/FreeSurfer_Longitudinal_Processing_brainmask/2009-08-06_13_22_42.0/I210238/brainmask.mgz").get_fdata()
    img = img.astype(np.uint8)
    log_3d(img, file_name="test/present/processed_mri")
    log_video(img, name="test/present/processed_mri_16")


def all_mri():
    df = pd.read_csv("dataset/all-T1.csv")
    print("Number of images: ", df.shape[0])
    print("Number of subjets: ", len(df['Subject ID'].unique()))

    img = read_image("dataset/MRI/ADNI/003_S_0908/MPRAGE/2013-01-23_08_22_46.0/I502287")
    img = np.maximum(img, 0)
    m = img.max()
    img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    log_3d(img, file_name="test/present/raw_mri")
    log_video(img, name="test/present/raw_mri_16")


def pet_processed():
    df = pd.read_csv("dataset/pet.csv")
    print("Number of images: ", df.shape[0])
    print("Number of subjets: ", len(df['Subject'].unique()))

    img = read_image(
        "dataset/PET/ADNI/341_S_6686/Coreg,_Avg,_Standardized_Image_and_Voxel_Size/2019-03-26_09_28_52.0/I1149014")
    img = np.maximum(img, 0)
    m = img.max()
    img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    log_3d(img, file_name="test/present/processed_pet")
    log_video(img, name="test/present/processed_pet_16")


def mri_preprocess():
    img = read_image("dataset/MRI/ADNI/941_S_7106/Accelerated_Sagittal_MPRAGE/2022-09-09_09_55_29.0/I1619403")
    img = np.transpose(img, (0, 2, 1))
    img = img[::-1, ::-1, ::-1]
    # img = np.maximum(img, 0)
    # m = img.max()
    # img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    # log_3d(img, file_name="test/present/mri-1")
    # log_video(img, name="test/present/mri-1")

    # skull_stripping(img)
    # img = nib.load("stripped.nii").get_fdata()
    # img = np.maximum(img, 0)
    # m = img.max()
    # img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    # log_3d(img, file_name="test/present/mri-2")
    # log_video(img, name="test/present/mri-2")

    # img = nib.load('template/stripped_cropped.nii').get_fdata()
    # img = np.maximum(img, 0)
    # m = img.max()
    # img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    # log_3d(img, file_name="test/present/mri-template")
    # log_video(img, name="test/present/mri-template")

    img = mri_registration("stripped.nii")
    img = np.maximum(img, 0)
    m = img.max()
    img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    log_3d(img, file_name="test/present/mri-3-SyN")
    log_video(img, name="test/present/mri-3-SyN")


def pet_preprocess():
    # img = read_image(
    #     "dataset/PET/ADNI/941_S_7041/Coreg,_Avg,_Standardized_Image_and_Voxel_Size/2022-03-09_10_28_29.0/I1557046")
    # img = np.transpose(img, (2, 1, 0))
    # img = img[::-1, ::-1, :]
    # img = np.maximum(img, 0)
    # m = img.max()
    # img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    # log_3d(img, file_name="test/present/pet-1")
    # log_video(img, name="test/present/pet-1")

    # img = read_image(
    #     "dataset/PET/ADNI/941_S_7041/Coreg,_Avg,_Standardized_Image_and_Voxel_Size/2022-03-09_10_28_29.0/I1557046")
    # img = img[:, 16:144, 16:144]
    # img = np.transpose(img, (2, 1, 0))
    # img = img[::-1, ::-1, :]
    # img = np.maximum(img, 0)
    # m = img.max()
    # img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    # log_3d(img, file_name="test/present/pet-2")
    # log_video(img, name="test/present/pet-2")

    # img = read_image(
    #     "dataset/PET/ADNI/941_S_7041/Coreg,_Avg,_Standardized_Image_and_Voxel_Size/2022-03-09_10_28_29.0/I1557046")
    # img = img[:, 16:144, 16:144]
    # img = np.transpose(img, (2, 1, 0))
    # img = img[::-1, ::-1, :]
    # skull_stripping(img)
    img = nib.load('stripped.nii').get_fdata()
    img = np.maximum(img, 0)
    m = img.max()
    img = np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)
    log_3d(img, file_name="test/present/pet-3")
    log_video(img, name="test/present/pet-3")


def run():
    # print("FreeSurfer Longitudinal Processing brainmask:")
    # mri_processed()
    # print("\nAll T1:")
    # all_mri()
    # print("\nPET:")
    # pet_processed()

    # print("MRI preprocessing")
    # mri_preprocess()

    print("PET preprocessing")
    pet_preprocess()
