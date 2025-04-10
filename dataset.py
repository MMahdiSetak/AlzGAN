import os
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime

import ants
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

group_mapping = {'CN': 0, 'MCI': 1, 'AD': 2}


def log_to_file_image(img, file_name):
    center_slices = [dim // 2 for dim in img.shape]
    # img = np.transpose(img, (0, 2, 1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # titles = ['Axial', 'Coronal', 'Sagittal']

    slices = [img[center_slices[0], :, :], img[:, center_slices[1], :], img[:, :, center_slices[2]]]

    for ax, slice_img in zip(axes, slices):
        ax.imshow(slice_img, cmap='gray')
        # ax.set_title(title)
        # ax.axis('off')

    # Save the figure to a file
    # plt.show()
    plt.savefig(f"log/pet/{file_name}.png")
    plt.close(fig)


def pair_log(mri, pet, filename):
    mri_centers = [dim // 2 for dim in mri.shape]
    pet_centers = [dim // 2 for dim in pet.shape]
    slices = [[mri[mri_centers[0], :, :], mri[:, mri_centers[1], :], mri[:, :, mri_centers[2]]],
              [pet[pet_centers[0], :, :], pet[:, pet_centers[1], :], pet[:, :, pet_centers[2]]]]
    _, axes = plt.subplots(2, 3, figsize=(4, 3))
    for i in range(2):
        for ax, slice_img in zip(axes[i], slices[i]):
            ax.imshow(np.rot90(slice_img), cmap='gray')
            ax.axis('off')
            ax.set_facecolor('none')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.savefig(f"log/pair/{filename}.png", transparent=True, bbox_inches='tight')
    # plt.show()
    plt.close()


def log_image(image):
    # Display the middle slice of the 3D volume using matplotlib
    mid_slice_index = image.shape[0] // 2
    plt.imshow(image[mid_slice_index], cmap='gray')
    plt.title('Brain MRI Image - Middle Slice')
    plt.axis('off')  # Turn off axis labels
    plt.show()

    # Optional: Display all slices one by one
    for i, slice in enumerate(image):
        plt.imshow(slice, cmap='gray', vmin=0, vmax=65535)
        plt.title(f'Brain MRI Image - Slice {i + 1}')
        plt.axis('off')
        plt.show()

        # print(f"Slice Instance Number: {slice.InstanceNumber}")
        # print("Patient Name:", slice.PatientName)
        # print("Patient ID:", slice.PatientID)
        # print("Study Date:", slice.StudyDate)
        # print("Modality:", slice.Modality)
        # print("Manufacturer:", slice.Manufacturer)
        # print(slice)
        # print('-' * 20)


# def dataset_info(paths):
#     max_intensity = 0
#     for path_to_datadir in paths:
#         print(path_to_datadir)
#         subjects = os.listdir(path_to_datadir)
#         with tqdm(total=len(subjects), desc="Subjects", leave=True) as pbar_subjects:
#             for subject in subjects:
#                 descs = os.listdir(f"{path_to_datadir}/{subject}")
#                 with tqdm(total=len(descs), desc=f"Descriptions ({subject})", leave=False) as pbar_descs:
#                     for desc in descs:
#                         dates = os.listdir(f"{path_to_datadir}/{subject}/{desc}")
#                         with tqdm(total=len(dates), desc=f"Dates ({subject}/{desc})", leave=False) as pbar_dates:
#                             for date in dates:
#                                 img_ids = os.listdir(f"{path_to_datadir}/{subject}/{desc}/{date}")
#                                 with tqdm(total=len(img_ids), desc=f"Images ({subject}/{desc}/{date})",
#                                           leave=False) as pbar_imgs:
#                                     for img_id in img_ids:
#                                         image_path = f"{path_to_datadir}/{subject}/{desc}/{date}/{img_id}"
#                                         files = [os.path.join(image_path, f) for f in os.listdir(image_path) if
#                                                  f.endswith('.dcm')]
#                                         slices = [pydicom.dcmread(f) for f in files]
#                                         slices.sort(key=lambda x: x.InstanceNumber)
#                                         image_3d = np.stack([s.pixel_array for s in slices])
#                                         max_intensity = max(image_3d.max(), max_intensity)
#                                         pbar_imgs.update(1)
#                                 pbar_dates.update(1)
#                     pbar_descs.update(1)
#                 pbar_subjects.update(1)
#
#     print(max_intensity)

def dataset_info(paths):
    rows = []
    # subjects = []
    # max_intensity = 0
    # shapes = defaultdict(int)
    # intensities = defaultdict(int)
    for path_to_datadir in tqdm(paths, leave=False):
        # print(path_to_datadir)
        subjects = os.listdir(path_to_datadir)
        for subject in tqdm(subjects, leave=False):
            # print(subject)
            descs = os.listdir(f"{path_to_datadir}/{subject}")
            for desc in descs:
                # print("\t", desc)
                dates = os.listdir(f"{path_to_datadir}/{subject}/{desc}")
                for date in tqdm(dates, leave=False):
                    # print("\t\t", date)
                    img_ids = os.listdir(f"{path_to_datadir}/{subject}/{desc}/{date}")
                    for img_id in img_ids:
                        # print("\t\t\t", img_id)
                        # start = time.time()
                        image_path = f"{path_to_datadir}/{subject}/{desc}/{date}/{img_id}"
                        # image_path = '2/2-MPRAGE/ADNI//012_S_4643/MPRAGE/2012-06-27_09_08_19.0/I313948'
                        files = [os.path.join(image_path, f) for f in os.listdir(image_path) if f.endswith('.dcm')]
                        if len(files) == 0:
                            print("empty directory: " + image_path)
                            continue
                        elif len(files) == 1:
                            dataset = pydicom.dcmread(files[0])
                            # slice_thickness = dataset.get((0x0018, 0x0088), None).value
                            slice_thickness = dataset.get((0x5200, 0x9230), None)[0].get((0x0028, 0x9110), None)[
                                0].SliceThickness
                            # dataset.data_element('BitsStored').value
                            bits_stored = dataset.get((0x0028, 0x0101), None).value
                            pixel_spacing = dataset.get((0x5200, 0x9230), None)[0].get((0x0028, 0x9110), None)[
                                0].PixelSpacing
                            image_3d = dataset.pixel_array

                        else:
                            slices = [pydicom.dcmread(f) for f in files]
                            slices.sort(key=lambda x: x.InstanceNumber)
                            slice_thickness = slices[0].SliceThickness
                            bits_stored = slices[0].BitsStored
                            pixel_spacing = slices[0].PixelSpacing
                            image_3d = np.stack([s.pixel_array for s in slices])
                        # log_image(image_3d)
                        # max_intensity = max(image_3d.max(), max_intensity)
                        # intensities[image_3d.max()] += 1
                        # shapes[image_3d.shape] += 1
                        try:
                            row_info = {
                                'path': path_to_datadir,
                                'subject': subject,
                                'imageID': img_id,
                                'shape': image_3d.shape,
                                'max_intensity': image_3d.max(),
                                'bits_stored': bits_stored,
                                'pixel_spacing': pixel_spacing,
                                'slice_thickness': slice_thickness
                            }
                            rows.append(row_info)
                        except AttributeError as e:
                            print(e)
                            print(image_path)
                    # df = pd.concat([df, pd.Series(row_info)], ignore_index=True)
    # print(subjects)
    # print(max_intensity)
    # print(intensities)
    # print(shapes)
    # print(np.array(list(shapes.values())).sum())
    columns = ['path', 'subject', 'imageID', 'shape', 'max_intensity', 'bits_stored', 'pixel_spacing',
               'slice_thickness']
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('mri_info.csv')


def read_image(path: str) -> np.ndarray:
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm')]
    slices = [pydicom.dcmread(f) for f in files]
    slices.sort(key=lambda x: x.InstanceNumber)
    image_3d = np.stack([s.pixel_array for s in slices])
    return image_3d


def calculate_subject_intersect(mri, pet):
    mri_subjects = set(os.listdir(mri))
    pet_subjects = set(os.listdir(pet))
    intersect_subjects = mri_subjects & pet_subjects
    return intersect_subjects


def resize_image(image, target_size):
    # Calculate the zoom factors for each dimension
    zoom_factors = [t / s for t, s in zip(target_size, image.shape)]
    # Resize the image
    resized_image = zoom(image, zoom_factors, order=3)  # order=3 for cubic interpolation
    return resized_image


mri_template = ants.image_read('template/icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.nii')


def mri_registration():
    moving_image = ants.image_read('stripped.nii')
    affine_registration = ants.registration(fixed=mri_template, moving=moving_image, type_of_transform='Affine')
    return affine_registration["warpedmovout"].numpy()


def pet_preprocess(img: np.ndarray) -> np.ndarray:
    img = np.transpose(img, (2, 1, 0))
    img = img[::-1, ::-1, :]
    skull_stripping(img)
    img = nib.load('stripped.nii').get_fdata()
    img = img[30:130, 10:150, :]
    normalized_img = normalize_image(img)
    # pet (160, 160, 96) -> (100, 140, 96)
    # mri (160, 200, 180)
    return normalized_img


MRI_ID_BLACKLIST = ['I1589895', 'I1591048', 'I1594002', 'I1611628', 'I1528880', 'I1557275', 'I1582497', 'I1623763',
                    'I1529590', 'I1340855', 'I297693', 'I368892', 'I421360', 'I377209', 'I313948', 'I1620655',
                    'I1327480', 'I1589927', 'I1327456', 'I1593283', 'I317117', 'I341919', 'I274422', 'I308385',
                    'I249403', 'I655568', 'I282005', 'I316545', 'I248517', 'I322057', 'I290413', 'I365244', 'I418180',
                    'I336709', 'I312872', 'I296878', 'I32421', 'I32853', 'I74064', 'I88487', 'I124701']


def mri_pet_label_info(mri_path, pet_path):
    damaged_img = 0
    total = 0
    intersect = calculate_subject_intersect(mri_path, pet_path)
    for subject in tqdm(intersect, leave=True):
        pet_dates_path = {}
        pet_descs = os.listdir(f"{pet_path}/{subject}")
        for pet_desc in pet_descs:
            pet_dates = os.listdir(f"{pet_path}/{subject}/{pet_desc}")
            for date in pet_dates:
                pet_dates_path[datetime.strptime(date,
                                                 '%Y-%m-%d_%H_%M_%S.%f')] = f"{pet_path}/{subject}/{pet_desc}/{date}"

        mri_descs = os.listdir(f"{mri_path}/{subject}")
        for mri_desc in tqdm(mri_descs, leave=False):
            mri_dates = os.listdir(f"{mri_path}/{subject}/{mri_desc}")
            for date in tqdm(mri_dates, leave=False):
                # start = time.time()
                mri_date = datetime.strptime(date, '%Y-%m-%d_%H_%M_%S.%f')
                closest_pet_date = min(pet_dates_path.keys(), key=lambda x: abs(x - mri_date))

                pet_img_id = os.listdir(pet_dates_path[closest_pet_date])[0]
                mri_img_id = os.listdir(f"{mri_path}/{subject}/{mri_desc}/{date}")[0]

                pet_img_path = f"{pet_dates_path[closest_pet_date]}/{pet_img_id}"
                mri_img_path = f"{mri_path}/{subject}/{mri_desc}/{date}/{mri_img_id}"

                if mri_img_id in MRI_ID_BLACKLIST:
                    damaged_img += 1
                    continue
                # analyzed = os.listdir("log/pet")
                # if f"{pet_img_id}.png" in analyzed:
                #     continue
                # pet_image = read_pet(pet_img_path)
                total += 1
                # log_to_file_image(pet_image, pet_img_id)
    print(total)
    print(damaged_img)


def create_mri_pet_label_dataset(mri_path, pet_path):
    intersect = calculate_subject_intersect(mri_path, pet_path)
    mri_target = (160, 200, 180)
    pet_target = (100, 140, 96)
    # num_imgs = 2656
    num_imgs = 2969
    indices = list(range(num_imgs))
    random.shuffle(indices)
    current_index = 0
    current_train_idx, current_val_idx, current_test_idx = 0, 0, 0
    df = pd.read_csv("mri_labels.csv")

    # Split indices into train (80%), validation (10%), and test (10%) sets
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    with h5py.File('mri_pet_label.hdf5', 'w') as h5f:
        mri_train_ds = h5f.create_dataset('mri_train', (len(train_indices), *mri_target), dtype='uint8')
        mri_val_ds = h5f.create_dataset('mri_val', (len(val_indices), *mri_target), dtype='uint8')
        mri_test_ds = h5f.create_dataset('mri_test', (len(test_indices), *mri_target), dtype='uint8')

        pet_train_ds = h5f.create_dataset('pet_train', (len(train_indices), *pet_target), dtype='uint8')
        pet_val_ds = h5f.create_dataset('pet_val', (len(val_indices), *pet_target), dtype='uint8')
        pet_test_ds = h5f.create_dataset('pet_test', (len(test_indices), *pet_target), dtype='uint8')

        label_train_ds = h5f.create_dataset('label_train', (len(train_indices),), dtype='int')
        label_val_ds = h5f.create_dataset('label_val', (len(val_indices),), dtype='int')
        label_test_ds = h5f.create_dataset('label_test', (len(test_indices),), dtype='int')

        for subject in tqdm(intersect, leave=True):
            pet_dates_path = {}
            pet_descs = os.listdir(f"{pet_path}/{subject}")
            for pet_desc in pet_descs:
                pet_dates = os.listdir(f"{pet_path}/{subject}/{pet_desc}")
                for date in pet_dates:
                    pet_dates_path[datetime.strptime(date,
                                                     '%Y-%m-%d_%H_%M_%S.%f')] = f"{pet_path}/{subject}/{pet_desc}/{date}"

            mri_descs = os.listdir(f"{mri_path}/{subject}")
            for mri_desc in tqdm(mri_descs, leave=False):
                mri_dates = os.listdir(f"{mri_path}/{subject}/{mri_desc}")
                for date in tqdm(mri_dates, leave=False):
                    # start = time.time()
                    mri_date = datetime.strptime(date, '%Y-%m-%d_%H_%M_%S.%f')
                    closest_pet_date = min(pet_dates_path.keys(), key=lambda x: abs(x - mri_date))

                    pet_img_id = os.listdir(pet_dates_path[closest_pet_date])[0]
                    mri_img_id = os.listdir(f"{mri_path}/{subject}/{mri_desc}/{date}")[0]

                    pet_img_path = f"{pet_dates_path[closest_pet_date]}/{pet_img_id}"
                    mri_img_path = f"{mri_path}/{subject}/{mri_desc}/{date}/{mri_img_id}"

                    if mri_img_id in MRI_ID_BLACKLIST:
                        continue
                    mri_image = read_image(mri_img_path)
                    mri_image = mri_preprocess(mri_image)
                    # mri_image = resize_image(preprocessed_mri, mri_target)

                    pet_image = read_image(pet_img_path)
                    pet_image = pet_preprocess(pet_image)
                    # pair_log(mri_image, pet_image, f'{mri_img_id}_{pet_img_id}')
                    # log_to_file_image(pet_image, pet_img_id)
                    # pet_image = resize_image(pet_image, pet_target)
                    # log_to_file_image(pet_image, pet_img_id)

                    label = df.loc[df['Image Data ID'] == mri_img_id].iloc[0]['Group']
                    label = group_mapping[label]

                    if current_index in train_indices:
                        mri_train_ds[current_train_idx] = mri_image
                        pet_train_ds[current_train_idx] = pet_image
                        label_train_ds[current_train_idx] = label
                        current_train_idx += 1
                    elif current_index in val_indices:
                        mri_val_ds[current_val_idx] = mri_image
                        pet_val_ds[current_val_idx] = pet_image
                        label_val_ds[current_val_idx] = label
                        current_val_idx += 1
                    elif current_index in test_indices:
                        mri_test_ds[current_test_idx] = mri_image
                        pet_test_ds[current_test_idx] = pet_image
                        label_test_ds[current_test_idx] = label
                        current_test_idx += 1
                    current_index += 1
                    # end = time.time()
                    # print(end - start)
            if current_index >= num_imgs:
                break

    print(current_index)


def info_analyze(csv_file):
    df = pd.read_csv(csv_file)
    new_dim = []
    old_dim = []
    indices_to_drop = []
    # thickness = defaultdict(int)
    for index, row in df.iterrows():
        shape = eval(row['shape'])
        pixel_zoom_factor = 1
        if shape[1] == 192 and shape[2] == 192:
            pixel_zoom_factor = 1
        elif shape[1] == 256:
            pixel_zoom_factor = 0.8
        else:
            print(shape)

        slice_zoom_factor = 1
        if shape[0] == 160:
            slice_zoom_factor = 1
        elif shape[0] > 160:
            slice_zoom_factor = 160 / shape[0]
        # elif shape[0] == 176 or shape[0] == 177:
        #     slice_zoom_factor = 0.91
        # elif shape[0] == 208:
        #     slice_zoom_factor = 0.769230769
        # elif shape[0] == 170:
        else:
            print(shape)
            indices_to_drop.append(index)
            continue
        # thickness[shape[0]] += 1

        thickness_zoom_factor = 1
        # if shape[0] == 192:

        # voxel_spacing = (row['slice_thickness'], *eval(row['pixel_spacing']))
        voxel_spacing = (slice_zoom_factor, pixel_zoom_factor, pixel_zoom_factor)
        # new_dim.append(tuple([round(a * b) for a, b in zip(shape, voxel_spacing)]))
        new_dim.append(
            tuple([round(a * b) for a, b in zip(shape, voxel_spacing)]))
        # old_dim.append(shape)
        old_dim.append((shape[1], shape[2]))

    df.drop(indices_to_drop, inplace=True)
    df['dimension'] = new_dim
    new_unique = list(set(new_dim))
    old_unique = list(set(old_dim))
    # print(thickness)
    return old_unique, new_unique

def scale_image(img: np.ndarray) -> np.ndarray | None:
    shape = img.shape
    # Determine slice zoom factor
    if shape[0] < 160:
        print(f"Unexpected slice dimension: {shape[0]}")
        return None
    slice_zoom_factor = 1 if shape[0] == 160 else 160 / shape[0]

    # Check for valid pixel dimensions
    if shape[2] not in (192, 256):
        print(f"Unexpected pixel dimensions: {shape[1]}, {shape[2]}")
        return None

    # Determine pixel zoom factor
    pixel_zoom_factor = 1 if shape[2] == 192 else 0.8

    # Scale the image using cubic interpolation
    scaled_image = zoom(img, (slice_zoom_factor, pixel_zoom_factor, pixel_zoom_factor), order=3)
    # scaled_image = zoom(img, (slice_zoom_factor/2, pixel_zoom_factor/2, pixel_zoom_factor/2), order=3)

    return scaled_image


def crop_image(img: np.ndarray) -> np.ndarray:
    crop_shape = (160, 192, 192)
    starts = [(dim - crop) // 2 for dim, crop in zip(img.shape, crop_shape)]
    return img[starts[0]:starts[0] + crop_shape[0],
           starts[1]:starts[1] + crop_shape[1],
           starts[2]:starts[2] + crop_shape[2]]


def normalize_image(img: np.ndarray) -> np.ndarray:
    # img = img.astype(np.float64) - img.min()
    img = np.maximum(img, 0)
    m = img.max()
    return ((img.astype(np.float64) * 255) / m).astype(np.uint8)


def mri_preprocess(img: np.ndarray) -> np.ndarray:
    img = np.transpose(img, (0, 2, 1))
    img = img[::-1, ::-1, ::-1]
    skull_stripping(img)
    img = mri_registration()
    # scaled_img = scale_image(img)
    # img = crop_image(img)
    img = img[15:175, 17:217, :180]
    normalized_img = normalize_image(img)
    return normalized_img


def create_mri_dataset(mri_path: str):
    mri_target = (160, 200, 180)
    # mri_target = (160, 192, 192)
    # mri_target = (160 // 2, 192 // 2, 192 // 2)
    num_imgs = 4575
    indices = list(range(num_imgs))
    random.shuffle(indices)
    current_train_idx, current_val_idx, current_test_idx = 0, 0, 0
    current_index = 0
    df = pd.read_csv("mri_labels.csv")

    # Split indices into train (80%), validation (10%), and test (10%) sets
    train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    with h5py.File('mri_label.hdf5', 'w') as h5f:
        mri_train_ds = h5f.create_dataset('mri_train', (len(train_indices), *mri_target), dtype='uint8')
        mri_val_ds = h5f.create_dataset('mri_val', (len(val_indices), *mri_target), dtype='uint8')
        mri_test_ds = h5f.create_dataset('mri_test', (len(test_indices), *mri_target), dtype='uint8')
        label_train_ds = h5f.create_dataset('label_train', (len(train_indices),), dtype='int')
        label_val_ds = h5f.create_dataset('label_val', (len(val_indices),), dtype='int')
        label_test_ds = h5f.create_dataset('label_test', (len(test_indices),), dtype='int')
        subjects = os.listdir(mri_path)
        for subject in tqdm(subjects, leave=False):
            descs = os.listdir(f"{mri_path}/{subject}")
            for desc in tqdm(descs, leave=False):
                dates = os.listdir(f"{mri_path}/{subject}/{desc}")
                for date in tqdm(dates, leave=False):
                    img_ids = os.listdir(f"{mri_path}/{subject}/{desc}/{date}")
                    for img_id in img_ids:
                        # start = time.time()
                        if img_id in MRI_ID_BLACKLIST:
                            continue
                        image_path = f"{mri_path}/{subject}/{desc}/{date}/{img_id}"
                        mri_image = read_image(image_path)
                        dataset_mri = mri_preprocess(mri_image)
                        # dataset_mri = resize_image(preprocessed_mri, mri_target)
                        label = df.loc[df['Image Data ID'] == img_id].iloc[0]['Group']
                        if current_index in train_indices:
                            mri_train_ds[current_train_idx] = dataset_mri
                            label_train_ds[current_train_idx] = group_mapping[label]
                            current_train_idx += 1
                        elif current_index in val_indices:
                            mri_val_ds[current_val_idx] = dataset_mri
                            label_val_ds[current_val_idx] = group_mapping[label]
                            current_val_idx += 1
                        elif current_index in test_indices:
                            mri_test_ds[current_test_idx] = dataset_mri
                            label_test_ds[current_test_idx] = group_mapping[label]
                            current_test_idx += 1
                        current_index += 1


affine = np.eye(4)


def skull_stripping(img):
    nifti_img = nib.Nifti1Image(img, affine)
    nib.save(nifti_img, "temp.nii")
    command = f'docker run --rm --gpus all -v .:/temp freesurfer/synthstrip:1.6 -i /temp/temp.nii -o /temp/stripped.nii'
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    if proc.returncode != 0:
        print(f'stripping failed: {proc.returncode}')
    # return nib.load("stripped.nii").get_fdata()


def pet_dcm2nii(pet_path):
    subjects = os.listdir(pet_path)
    shapes = defaultdict(int)
    for subject in tqdm(subjects, leave=True):
        sub_path = f"{pet_path}/{subject}/Coreg,_Avg,_Standardized_Image_and_Voxel_Size"
        pet_dates = os.listdir(sub_path)
        for date in tqdm(pet_dates, leave=False):
            img_id = os.listdir(f"{sub_path}/{date}")[0]
            image_path = f"{sub_path}/{date}/{img_id}"
            pet_image = read_image(image_path)
            pet_image = pet_preprocess(pet_image)
            shapes[pet_image.shape] += 1
            log_to_file_image(pet_image, img_id)
    print(shapes)


def mri_dcm2nii(mri_path):
    count = 0
    subjects = os.listdir(mri_path)
    for subject in tqdm(subjects, leave=False):
        descs = os.listdir(f"{mri_path}/{subject}")
        for desc in tqdm(descs, leave=False):
            dates = os.listdir(f"{mri_path}/{subject}/{desc}")
            for date in tqdm(dates, leave=False):
                img_ids = os.listdir(f"{mri_path}/{subject}/{desc}/{date}")
                for img_id in img_ids:
                    image_path = f"{mri_path}/{subject}/{desc}/{date}/{img_id}"
                    try:
                        mri_image = read_image(image_path)
                        mri_image = mri_preprocess(mri_image)
                        log_to_file_image(mri_image, img_id)
                        count += 1
                    except Exception as e:
                        print(e)
                        print(f'failed on {img_id}')
    print(count)


mri_data_paths = "MRI/ADNI/"
pet_data_path = "PET/ADNI/"

# dataset_info(mri_data_paths)
# create_mri_pet_label_dataset(mri_data_paths, pet_data_path)
# mri_pet_label_info(mri_data_paths, pet_data_path)
# mri_dcm2nii(mri_data_paths)
# pet_dcm2nii(pet_data_path)

# info_analyze('mri_info.csv')

# create_mri_dataset(mri_data_paths)
