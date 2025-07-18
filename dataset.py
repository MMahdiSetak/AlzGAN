import os
import random
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta

import ants
import h5py
import imageio
import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# group_mapping = {'CN': 0, 'SMC': 1, 'EMCI': 2, 'MCI': 3, 'LMCI': 4, 'AD': 5}
group_mapping = {'CN': 0, 'MCI': 1, 'AD': 2}
reversed_group_mapping = {0: 'CN', 1: 'MCI', 2: 'AD'}


def log_to_file_image(img, title="", file_name='test'):
    center_slices = [dim // 2 for dim in img.shape]
    # img = np.transpose(img, (0, 2, 1))

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    # titles = ['Axial', 'Coronal', 'Sagittal']

    slices = [img[center_slices[0], :, :], img[:, center_slices[1], :], img[:, :, center_slices[2]]]

    for ax, slice_img in zip(axes, slices):
        ax.imshow(slice_img, cmap='gray')
        # ax.set_title(title)
        # ax.axis('off')

    # Save the figure to a file
    # plt.show()
    plt.savefig(f"log/mri/{file_name}.png")
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
    plt.savefig(f"log/2d/pair/{filename}.png", transparent=True, bbox_inches='tight')
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

    #     print(f"Slice Instance Number: {slice.InstanceNumber}")
    #     print("Patient Name:", slice.PatientName)
    #     print("Patient ID:", slice.PatientID)
    #     print("Study Date:", slice.StudyDate)
    #     print("Modality:", slice.Modality)
    #     print("Manufacturer:", slice.Manufacturer)
    #     print(slice)
    #     print('-' * 20)


def log_video(img, name='pet_scan_video'):
    # pet_scan_clipped = np.clip(img, 0, None)
    #
    # # Find the maximum value for scaling
    # max_val = pet_scan_clipped.max()
    #
    # # Scale to 0-255 and convert to uint8; handle case where max_val is 0
    # if max_val > 0:
    #     scaled = (pet_scan_clipped / max_val * 255).astype(np.uint8)
    # else:
    #     scaled = np.zeros_like(pet_scan_clipped, dtype=np.uint8)

    # Initialize video writer: 'mp4v' codec, 10 fps, 160x160 frame size
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('pet_scan_video.mp4', fourcc, 10, (160, 160))
    #
    # # Write each slice as a frame
    # for i in range(96):
    #     # Extract the 2D slice (160, 160)
    #     slice_gray = img[i, :, :]
    #     # Convert grayscale to BGR for OpenCV video compatibility
    #     slice_bgr = cv2.cvtColor(slice_gray, cv2.COLOR_GRAY2BGR)
    #     # Write frame to video
    #     out.write(slice_bgr)
    #
    # # Release the video writer to finalize the file
    # out.release()
    H, W, Z = img.shape
    writer = imageio.get_writer(f'{name}.mp4', fps=8, codec='libx264')
    for w in range(W):
        slice_ = img[:, w, :]
        # Normalize to [0,255] uint8
        # img8 = ((slice_ - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
        writer.append_data(slice_)
    writer.close()


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


def read_mri(path: str) -> np.ndarray:
    img = nib.load(f"{path}/brainmask.mgz")
    img = img.get_fdata().astype(np.uint8)
    return img


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


# mri_template = ants.image_read('template/icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.nii')
mri_template = ants.image_read('template/stripped_cropped.nii')


def mri_registration(path):
    # moving_image = ants.image_read('stripped.nii')
    moving_image = ants.image_read(path)
    # TODO Add N4
    # corrected_image = ants.n4_bias_field_correction(moving_image, shrink_factor=4, convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7})
    # TODO SyN pipline
    registration = ants.registration(fixed=mri_template, moving=moving_image, type_of_transform='Rigid')
    return registration["warpedmovout"].numpy()


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


def pet_preprocess2(img: np.ndarray) -> np.ndarray:
    img = img[:, 16:144, 16:144]
    img = np.transpose(img, (2, 1, 0))
    img = img[::-1, ::-1, :]
    skull_stripping(img)
    img = nib.load('stripped.nii').get_fdata()
    # img = img[16:112, :, :]
    normalized_img = normalize_image(img)
    # padded_image = np.pad(normalized_img, ((0, 0), (0, 0), (16, 16)), mode='constant')
    # pet (160, 160, 96) -> (96, 128, 96) -> (128, 128, 96)
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


def count_pair_images(subjects, mri_path, pet_path):
    valid_count = count = 0
    date_distance = []
    for subject in subjects:
        pet_dates_path = {}
        pet_descs = os.listdir(f"{pet_path}/{subject}")
        for pet_desc in pet_descs:
            pet_dates = os.listdir(f"{pet_path}/{subject}/{pet_desc}")
            for date in pet_dates:
                pet_dates_path[datetime.strptime(date,
                                                 '%Y-%m-%d_%H_%M_%S.%f')] = f"{pet_path}/{subject}/{pet_desc}/{date}"
        mri_descs = os.listdir(f"{mri_path}/{subject}")
        for mri_desc in mri_descs:
            mri_dates = os.listdir(f"{mri_path}/{subject}/{mri_desc}")
            for date in mri_dates:
                mri_date = datetime.strptime(date, '%Y-%m-%d_%H_%M_%S.%f')
                closest_pet_date = min(pet_dates_path.keys(), key=lambda x: abs(x - mri_date))
                # date_distance.append((mri_date, closest_pet_date))

                mri_img_id = os.listdir(f"{mri_path}/{subject}/{mri_desc}/{date}")[0]
                if mri_img_id in MRI_ID_BLACKLIST:
                    continue
                distance_days = abs((mri_date - closest_pet_date).days)
                distance_months = distance_days / 30  # Convert days to months
                date_distance.append(distance_months)
                if distance_days < 180:
                    valid_count += 1
                count += 1

    # Plotting the histogram
    # bins = np.arange(0, max(date_distance) + 1, 1)  # 6-month intervals
    # bins = np.arange(0, 48, 6)  # 6-month intervals
    # plt.hist(date_distance, bins=bins, edgecolor='black')
    # plt.xlabel('Distance in months')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of MRI-PET Date Distances')
    # plt.savefig('mri_pet_histogram.png')  # Save the plot instead of showing it
    # plt.show()  # Save the plot instead of showing it

    return valid_count


def create_mri_pet_label_dataset(mri_path, pet_path):
    intersect = calculate_subject_intersect(mri=mri_path, pet=pet_path)
    train_subj, val_subj, test_subj = split_subject(list(intersect))
    mri_target = (160, 192, 160)
    pet_target = (128, 128, 96)
    subj_split = {'train': train_subj, 'val': val_subj, 'test': test_subj}
    split_num = {
        'train': count_pair_images(train_subj, mri_path=mri_path, pet_path=pet_path),
        'val': count_pair_images(val_subj, mri_path=mri_path, pet_path=pet_path),
        'test': count_pair_images(test_subj, mri_path=mri_path, pet_path=pet_path),
    }
    # df = pd.read_csv("mri_labels.csv")
    # df = pd.read_csv("dataset/mri.csv")
    # TODO float32
    with h5py.File('mri_pet_label_v5.1_Rigid.hdf5', 'w') as h5f:
        ds = {
            'mri_train': h5f.create_dataset('mri_train', (split_num['train'], *mri_target), dtype='uint8'),
            'mri_val': h5f.create_dataset('mri_val', (split_num['val'], *mri_target), dtype='uint8'),
            'mri_test': h5f.create_dataset('mri_test', (split_num['test'], *mri_target), dtype='uint8'),

            'pet_train': h5f.create_dataset('pet_train', (split_num['train'], *pet_target), dtype='uint8'),
            'pet_val': h5f.create_dataset('pet_val', (split_num['val'], *pet_target), dtype='uint8'),
            'pet_test': h5f.create_dataset('pet_test', (split_num['test'], *pet_target), dtype='uint8'),

            # 'label_train': h5f.create_dataset('label_train', (split_num['train'],), dtype='int'),
            # 'label_val': h5f.create_dataset('label_val', (split_num['val'],), dtype='int'),
            # 'label_test': h5f.create_dataset('label_test', (split_num['test'],), dtype='int'),
        }

        for split, subjects in tqdm(subj_split.items(), leave=False):
            indices = list(range(split_num[split]))
            random.shuffle(indices)
            current_index = 0
            for subject in tqdm(subjects, leave=True):
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
                        mri_date = datetime.strptime(date, '%Y-%m-%d_%H_%M_%S.%f')
                        closest_pet_date = min(pet_dates_path.keys(), key=lambda x: abs(x - mri_date))
                        distance_days = abs((mri_date - closest_pet_date).days)
                        if distance_days >= 180:
                            continue

                        pet_img_id = os.listdir(pet_dates_path[closest_pet_date])[0]
                        mri_img_id = os.listdir(f"{mri_path}/{subject}/{mri_desc}/{date}")[0]

                        pet_img_path = f"{pet_dates_path[closest_pet_date]}/{pet_img_id}"
                        mri_img_path = f"{mri_path}/{subject}/{mri_desc}/{date}/{mri_img_id}"

                        # if mri_img_id in MRI_ID_BLACKLIST:
                        #     continue

                        # mri_image = read_image(mri_img_path)
                        # mri_image = mri_preprocess(mri_image)
                        # mri_image = read_mri(mri_img_path)
                        mri_image = mri_preprocess2(f"{mri_img_path}/brainmask.mgz")

                        pet_image = read_image(pet_img_path)
                        pet_image = pet_preprocess2(pet_image)

                        pair_log(mri_image, pet_image, mri_img_id)

                        # label = df.loc[df['Image Data ID'] == mri_img_id].iloc[0]['Group']
                        ds[f'mri_{split}'][indices[current_index]] = mri_image
                        ds[f'pet_{split}'][indices[current_index]] = pet_image
                        # ds[f'label_{split}'][indices[current_index]] = group_mapping[label]
                        current_index += 1

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


# TODO z-score normalization (mean=0, std=1)
def normalize_image(img: np.ndarray) -> np.ndarray:
    # img = img.astype(np.float64) - img.min()
    img = np.maximum(img, 0)
    m = img.max()
    # TODO img.astype(np.float32)
    return np.round((img.astype(np.float64) * 255) / m).astype(np.uint8)


def mri_preprocess(img: np.ndarray) -> np.ndarray:
    img = np.transpose(img, (0, 2, 1))
    img = img[::-1, ::-1, ::-1]
    skull_stripping(img)
    img = mri_registration()
    # scaled_img = scale_image(img)
    # img = crop_image(img)
    # img = img[15:175, 17:217, :180]
    img = img[:-1, 18:210, :-1]
    img = resize_image(img, (128, 128, 128))
    normalized_img = normalize_image(img)
    return normalized_img


def mri_preprocess2(path: str) -> np.ndarray:
    img = mri_registration(path)
    # img = img[15:175, 20:212, 5:165]
    normalized_img = normalize_image(img)
    return normalized_img


def split_subject(subjects: [str]):
    # Split indices into train (80%), validation (10%), and test (10%) sets
    train_indices, temp_indices = train_test_split(subjects, test_size=0.2, random_state=42)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)
    return train_indices, val_indices, test_indices


def count_subject_image(subjects, mri_path: str):
    count = 0
    dgs = pd.read_csv("dataset/DXSUM.csv").dropna(subset=['DIAGNOSIS'])
    dgs['EXAMDATE'] = pd.to_datetime(dgs['EXAMDATE'])
    subject_rows_cache = {subject: dgs[dgs['PTID'] == subject].copy() for subject in subjects}
    for subject in subjects:
        rows = subject_rows_cache.get(subject)
        descs = os.listdir(f"{mri_path}/{subject}")
        for desc in descs:
            dates = os.listdir(f"{mri_path}/{subject}/{desc}")
            for date in dates:
                acc_date = datetime.strptime(date, '%Y-%m-%d_%H_%M_%S.%f')
                img_ids = os.listdir(f"{mri_path}/{subject}/{desc}/{date}")
                for img_id in img_ids:
                    if img_id in MRI_ID_BLACKLIST:
                        continue
                    rows['time_diff'] = (rows['EXAMDATE'] - acc_date).abs()
                    nearest_row_index = rows['time_diff'].idxmin()
                    min_time_diff = rows.loc[nearest_row_index, 'time_diff']
                    if min_time_diff > timedelta(days=180):
                        continue
                    count += 1
    return count


def create_mri_dataset(mri_path: str):
    mri_target = (160, 192, 160)
    subjects = os.listdir(mri_path)
    # subjects = os.listdir(mri_path)[:10]
    train_subj, val_subj, test_subj = split_subject(subjects)
    subj_split = {'train': train_subj, 'val': val_subj, 'test': test_subj}
    split_num = {
        'train': count_subject_image(train_subj, mri_path),
        'val': count_subject_image(val_subj, mri_path),
        'test': count_subject_image(test_subj, mri_path),
    }
    # df = pd.read_csv("dataset/mri.csv")
    dgs = pd.read_csv("dataset/DXSUM.csv").dropna(subset=['DIAGNOSIS'])
    dgs['EXAMDATE'] = pd.to_datetime(dgs['EXAMDATE'])
    # wrong_labels = 0
    with h5py.File('mri_label_v5.1_Rigid.hdf5', 'w') as h5f:
        ds = {
            'mri_train': h5f.create_dataset('mri_train', (split_num['train'], *mri_target), dtype='uint8'),
            'mri_val': h5f.create_dataset('mri_val', (split_num['val'], *mri_target), dtype='uint8'),
            'mri_test': h5f.create_dataset('mri_test', (split_num['test'], *mri_target), dtype='uint8'),
            'label_train': h5f.create_dataset('label_train', (split_num['train'],), dtype='int'),
            'label_val': h5f.create_dataset('label_val', (split_num['val'],), dtype='int'),
            'label_test': h5f.create_dataset('label_test', (split_num['test'],), dtype='int')
        }
        for split, subjects in tqdm(subj_split.items(), leave=False):
            indices = list(range(split_num[split]))
            random.shuffle(indices)
            current_index = 0
            subject_rows_cache = {subject: dgs[dgs['PTID'] == subject].copy() for subject in subjects}
            for subject in tqdm(subjects, leave=False):
                descs = os.listdir(f"{mri_path}/{subject}")
                rows = subject_rows_cache.get(subject)
                for desc in tqdm(descs, leave=False):
                    dates = os.listdir(f"{mri_path}/{subject}/{desc}")
                    for date in tqdm(dates, leave=False):
                        acc_date = datetime.strptime(date, '%Y-%m-%d_%H_%M_%S.%f')
                        img_ids = os.listdir(f"{mri_path}/{subject}/{desc}/{date}")
                        for img_id in img_ids:
                            if img_id in MRI_ID_BLACKLIST:
                                continue
                            rows['time_diff'] = (rows['EXAMDATE'] - acc_date).abs()
                            nearest_row_index = rows['time_diff'].idxmin()
                            min_time_diff = rows.loc[nearest_row_index, 'time_diff']
                            new_label = int(rows.loc[nearest_row_index, 'DIAGNOSIS'] - 1)
                            if min_time_diff > timedelta(days=180):
                                continue
                            image_path = f"{mri_path}/{subject}/{desc}/{date}/{img_id}"
                            # mri_image = read_image(image_path)
                            # mri_image = read_mri(image_path)
                            # mri_image = np.transpose(mri_image, (0, 2, 1))
                            # mri_image = mri_image[40:220, 30:230, 190:30:-1]
                            # log_to_file_image(mri_image, file_name=img_id)
                            dataset_mri = mri_preprocess2(f"{image_path}/brainmask.mgz")
                            log_to_file_image(dataset_mri, title=reversed_group_mapping[new_label], file_name=img_id)
                            # old_label = df.loc[df['Image Data ID'] == img_id].iloc[0]['Group']
                            # old_label = group_mapping[old_label]
                            # if old_label != new_label:
                            #     print(old_label, new_label)
                            #     wrong_labels += 1
                            ds[f'mri_{split}'][indices[current_index]] = dataset_mri
                            ds[f'label_{split}'][indices[current_index]] = new_label
                            current_index += 1
    # print(wrong_labels)


def create_pet_dataset(pet_path: str):
    pet_target = (128, 128, 128)
    subjects = os.listdir(pet_path)
    train_subj, val_subj, test_subj = split_subject(subjects)
    subj_split = {'train': train_subj, 'val': val_subj, 'test': test_subj}
    split_num = {
        'train': count_subject_image(train_subj, pet_path),
        'val': count_subject_image(val_subj, pet_path),
        'test': count_subject_image(test_subj, pet_path),
    }
    df = pd.read_csv("dataset/pet.csv")

    with h5py.File('pet_label.hdf5', 'w') as h5f:
        ds = {
            'pet_train': h5f.create_dataset('pet_train', (split_num['train'], *pet_target), dtype='uint8'),
            'pet_val': h5f.create_dataset('pet_val', (split_num['val'], *pet_target), dtype='uint8'),
            'pet_test': h5f.create_dataset('pet_test', (split_num['test'], *pet_target), dtype='uint8'),
            'label_train': h5f.create_dataset('label_train', (split_num['train'],), dtype='int'),
            'label_val': h5f.create_dataset('label_val', (split_num['val'],), dtype='int'),
            'label_test': h5f.create_dataset('label_test', (split_num['test'],), dtype='int')
        }
        for split, subjects in tqdm(subj_split.items(), leave=False):
            indices = list(range(split_num[split]))
            random.shuffle(indices)
            current_index = 0
            for subject in tqdm(subjects, leave=False):
                descs = os.listdir(f"{pet_path}/{subject}")
                for desc in tqdm(descs, leave=False):
                    dates = os.listdir(f"{pet_path}/{subject}/{desc}")
                    for date in tqdm(dates, leave=False):
                        img_ids = os.listdir(f"{pet_path}/{subject}/{desc}/{date}")
                        for img_id in img_ids:
                            image_path = f"{pet_path}/{subject}/{desc}/{date}/{img_id}"
                            pet_image = read_image(image_path)
                            # log_video(pet_image)
                            dataset_pet = pet_preprocess2(pet_image)
                            # log_video(dataset_pet)
                            # log_to_file_image(dataset_pet)
                            label = df.loc[df['Image Data ID'] == img_id].iloc[0]['Group']
                            ds[f'pet_{split}'][indices[current_index]] = dataset_pet
                            ds[f'label_{split}'][indices[current_index]] = group_mapping[label]
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


mri_data_path = "dataset/MRI2/ADNI/"
pet_data_path = "dataset/PET/ADNI/"

# dataset_info(mri_data_paths)
# create_mri_pet_label_dataset(mri_data_paths, pet_data_path)
# mri_pet_label_info(mri_data_paths, pet_data_path)
# mri_dcm2nii(mri_data_paths)
# pet_dcm2nii(pet_data_path)

# info_analyze('mri_info.csv')

# create_mri_dataset(mri_data_paths)
