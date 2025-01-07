import os
import random
import time

import ants
import h5py
import numpy as np
import nibabel as nib
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pydicom
from collections import defaultdict
from datetime import datetime
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

group_mapping = {'CN': 0, 'MCI': 1, 'AD': 2}


def log_to_file_image(img, file_name):
    center_slices = [dim // 2 for dim in img.shape]
    # img = np.transpose(img, (0, 2, 1))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Axial', 'Coronal', 'Sagittal']

    slices = [img[center_slices[0], :, :], img[:, center_slices[1], :], img[:, :, center_slices[2]]]

    for ax, slice_img, title in zip(axes, slices, titles):
        ax.imshow(slice_img, cmap='gray')
        ax.set_title(title)
        # ax.axis('off')

    # Save the figure to a file
    # plt.show()
    plt.savefig(f"log/pet/{file_name}.png")
    plt.close(fig)


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


def read_3d_image(path: str) -> np.ndarray | None:
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm')]
    if len(files) == 0:
        img_path = f'{path}/{os.listdir(path)[0]}'
        if img_path.endswith('.v'):
            try:
                img = nib.ecat.load(img_path)
            except Exception:
                return None
        elif img_path.endswith('.hdr'):
            try:
                img = nib.load(img_path)
            except nib.filebasedimages.ImageFileError:
                return None
        else:
            return None
        image_3d = img.get_fdata()[:, :, :, 0]
        image_3d = np.transpose(image_3d, (2, 0, 1))
        return image_3d
    elif len(files) == 1:
        image_3d = pydicom.dcmread(files[0]).pixel_array
    else:
        slices = [pydicom.dcmread(f) for f in files]
        slices.sort(key=lambda x: x.InstanceNumber)
        image_3d = np.stack([s.pixel_array for s in slices])
        if slices[0].Modality == 'PT':
            image_3d = image_3d[:slices[0].NumberOfSlices]
    return image_3d


def read_pet(path: str) -> np.ndarray:
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


# Load the MNI template and the subject's MRI scan
pet_mni_template_path = "template/MNI152_PET_1mm.nii"
fixed_image = ants.image_read(pet_mni_template_path)


def pet_registration(moving_image_path):
    # log_to_file_image(fixed_image.numpy(), "debug")
    moving_image = ants.from_numpy(moving_image_path)
    # affine_registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Rigid')
    # affine_registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Affine')
    # affine_registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyNRA')
    affine_registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='TRSAA')
    # affine_registration = ants.registration(fixed=fixed_image, moving=moving_image,
    #                                         type_of_transform='antsRegistrationSyN[a]')
    return affine_registration["warpedmovout"].numpy()


def pet_image_preprocess(img: np.ndarray) -> np.ndarray:
    # scaled_img = scale_image(img)
    # cropped_img = crop_image(scaled_img)
    normalized_img = normalize_image(img)
    # normalized_img = np.transpose(normalized_img, (2, 1, 0))
    # normalized_img = normalized_img[:, ::-1, ::-1]
    # log_to_file_image(normalized_img, "debug")
    registered_img = pet_registration(normalized_img)
    return registered_img


def mri_pet_label_info(mri_path, pet_path):
    image_id_black_list = ['I32421', 'I32853']
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

                if mri_img_id in image_id_black_list:
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
    damaged_img = 0
    mri_target = (160, 192, 192)
    pet_target = (35, 128, 128)
    num_imgs = 2656
    # num_imgs = 80
    indices = list(range(num_imgs))
    random.shuffle(indices)
    current_index = 0
    # current_train_idx, current_val_idx, current_test_idx = 0, 0, 0
    image_id_black_list = ['I32421', 'I32853']
    # df = pd.read_csv("mri_labels.csv")

    # Split indices into train (80%), validation (10%), and test (10%) sets
    # train_indices, temp_indices = train_test_split(indices, test_size=0.2, random_state=42)
    # val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

    with h5py.File('mri_pet_label_large_test.hdf5', 'w') as h5f:
        # mri_train_ds = h5f.create_dataset('mri_train', (len(train_indices), *mri_target), dtype='uint8')
        # mri_val_ds = h5f.create_dataset('mri_val', (len(val_indices), *mri_target), dtype='uint8')
        # mri_test_ds = h5f.create_dataset('mri_test', (len(test_indices), *mri_target), dtype='uint8')
        #
        # pet_train_ds = h5f.create_dataset('pet_train', (len(train_indices), *pet_target), dtype='uint8')
        # pet_val_ds = h5f.create_dataset('pet_val', (len(val_indices), *pet_target), dtype='uint8')
        # pet_test_ds = h5f.create_dataset('pet_test', (len(test_indices), *pet_target), dtype='uint8')
        #
        # label_train_ds = h5f.create_dataset('label_train', (len(train_indices),), dtype='int')
        # label_val_ds = h5f.create_dataset('label_val', (len(val_indices),), dtype='int')
        # label_test_ds = h5f.create_dataset('label_test', (len(test_indices),), dtype='int')

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
                    # if pet_img_id != 'I202949':  # I1430127 I82843 -> flip I1295819 I1480309-> dont flip
                    #     continue

                    if mri_img_id in image_id_black_list:
                        continue
                    analyzed = os.listdir("log/pet")
                    if f"{pet_img_id}.png" in analyzed:
                        continue
                    # mri_image = read_3d_image(mri_img_path)
                    # if mri_image.shape[0] < 160 or mri_image.shape[1] not in (192, 256):
                    #     damaged_img += 1
                    #     continue
                    # preprocessed_mri = mri_image_preprocess(mri_image)
                    # mri_image = resize_image(preprocessed_mri, mri_target)
                    # pet_img_path = 'PET/ADNI//022_S_4266/ADNI_Brain_PET__Raw_FDG/2011-12-20_11_02_13.0/I274741'
                    # pet_img_path = 'PET/ADNI//024_S_6033/Dy1_[F-18]FDG_4i_16s/2017-07-10_08_14_03.0/I872299' # ASC -> ACS

                    pet_image = read_pet(pet_img_path)
                    if pet_image is None:
                        damaged_img += 1
                        continue
                    log_to_file_image(pet_image, pet_img_id)
                    # pet_image = pet_image_preprocess(pet_image)
                    # log_to_file_image(pet_image, pet_img_id)
                    # pet_image = resize_image(pet_image, pet_target)
                    # log_to_file_image(pet_image, pet_img_id)
                    # if pet_image.max() < 1:
                    #     print("oh no empy pet")

                    # label = df.loc[df['Image Data ID'] == mri_img_id].iloc[0]['Group']
                    # label = group_mapping[label]

                    # if current_index in train_indices:
                    #     mri_train_ds[current_train_idx] = mri_image
                    #     pet_train_ds[current_train_idx] = pet_image
                    #     label_train_ds[current_train_idx] = label
                    #     current_train_idx += 1
                    # elif current_index in val_indices:
                    #     mri_val_ds[current_val_idx] = mri_image
                    #     pet_val_ds[current_val_idx] = pet_image
                    #     label_val_ds[current_val_idx] = label
                    #     current_val_idx += 1
                    # elif current_index in test_indices:
                    #     mri_test_ds[current_test_idx] = mri_image
                    #     pet_test_ds[current_test_idx] = pet_image
                    #     label_test_ds[current_test_idx] = label
                    #     current_test_idx += 1
                    current_index += 1
                    # end = time.time()
                    # print(end - start)
            if current_index >= num_imgs:
                break

    print(damaged_img)
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
    if shape[1] not in (192, 256):
        print(f"Unexpected pixel dimensions: {shape[1]}, {shape[2]}")
        return None

    # Determine pixel zoom factor
    pixel_zoom_factor = 1 if shape[1] == 192 else 0.8

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


def mri_image_preprocess(img: np.ndarray) -> np.ndarray:
    scaled_img = scale_image(img)
    cropped_img = crop_image(scaled_img)
    normalized_img = normalize_image(cropped_img)
    return normalized_img


def create_mri_dataset(mri_path: str):
    # mri_target = (160, 192, 192)
    mri_target = (160 // 2, 192 // 2, 192 // 2)
    num_imgs = 4608
    image_id_black_list = ['I32421', 'I32853']
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
                        if img_id in image_id_black_list:
                            continue
                        image_path = f"{mri_path}/{subject}/{desc}/{date}/{img_id}"
                        # if os.path.exists(f"log/{img_id}.png"):
                        #     continue
                        mri_image = read_3d_image(image_path)
                        if mri_image.shape[0] < 160 or mri_image.shape[1] not in (192, 256):
                            # print(mri_image.shape)
                            continue
                        preprocessed_mri = mri_image_preprocess(mri_image)
                        dataset_mri = resize_image(preprocessed_mri, mri_target)
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
                        # mri_ds[indices[current_index]] = dataset_mri
                        # label = df.loc[df['Image Data ID'] == img_id].iloc[0]['Group']
                        # label_ds[indices[current_index]] = group_mapping[label]
                        current_index += 1
                        # log_to_file_image(preprocessed_mri, img_id)


def pet_dcm2nii(pet_path):
    subjects = os.listdir(pet_path)
    shapes = defaultdict(int)
    for subject in tqdm(subjects, leave=True):
        sub_path = f"{pet_path}/{subject}/Coreg,_Avg,_Standardized_Image_and_Voxel_Size"
        pet_dates = os.listdir(sub_path)
        for date in tqdm(pet_dates, leave=False):
            img_id = os.listdir(f"{sub_path}/{date}")[0]
            image_path = f"{sub_path}/{date}/{img_id}"
            pet_image = read_pet(image_path)
            pet_image = np.transpose(pet_image, (2, 1, 0))
            pet_image = pet_image[::-1, ::-1, :]
            affine = np.eye(4)
            nifti_img = nib.Nifti1Image(pet_image, affine)
            # Save as NIfTI file
            nib.save(nifti_img, "output_file.nii")
            shapes[pet_image.shape] += 1
            # log_to_file_image(pet_image, img_id)
    print(shapes)


mri_data_paths = "MRI/ADNI/"
pet_data_path = "PET/ADNI/"

# dataset_info(mri_data_paths)
# create_mri_pet_label_dataset(mri_data_paths, pet_data_path)
# mri_pet_label_info(mri_data_paths, pet_data_path)
pet_dcm2nii(mri_data_paths)

# info_analyze('mri_info.csv')

# create_mri_dataset(mri_data_paths)
