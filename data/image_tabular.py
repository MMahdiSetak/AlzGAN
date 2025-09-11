import os
import tempfile

import ants
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from data.image import read_image, pet_preprocess2


def merge_mri_csv(mri_path: str):
    df_tb = pd.read_csv('dataset/tabular/all.csv')
    df_tb['image_path'] = pd.NA
    df_img = pd.read_csv('dataset/mri.csv')
    subjects = os.listdir(mri_path)
    total = cnt = 0
    for subject in tqdm(subjects, leave=False):
        descs = os.listdir(f"{mri_path}/{subject}")
        for desc in tqdm(descs, leave=False):
            dates = os.listdir(f"{mri_path}/{subject}/{desc}")
            for date in tqdm(dates, leave=False):
                img_ids = os.listdir(f"{mri_path}/{subject}/{desc}/{date}")
                for img_id in img_ids:
                    img_visit = df_img.loc[df_img['Image Data ID'] == img_id]['Visit'].iloc[0]
                    tb_row_idx = df_tb.loc[(df_tb['PTID'] == subject) & (df_tb['VISCODE2'] == img_visit)].index
                    if len(tb_row_idx) != 1:
                        cnt += 1
                        total += 1
                        continue
                    df_tb.loc[tb_row_idx, 'image_path'] = os.path.join(mri_path, subject, desc, date, img_id)
                    total += 1
    print(f"{cnt}/{total}")

    df_tb = df_tb.dropna(subset=['image_path'])
    df_tb['DIAGNOSIS'] = df_tb['DIAGNOSIS'] - 1

    output_path = 'dataset/tabular/img_merged.csv'
    df_tb.to_csv(output_path, index=False)
    print(f"Updated DataFrame saved to {output_path}")


def merge_mri_csv_pet(mri_path: str, pet_path: str):
    df_tb = pd.read_csv('dataset/tabular/all.csv')
    df_tb['mri_path'] = pd.NA
    df_tb['pet_path'] = pd.NA
    df_mri = pd.read_csv('dataset/mri.csv')
    df_pet = pd.read_csv('dataset/pet.csv')
    subjects = os.listdir(mri_path)
    total = cnt = 0
    for subject in tqdm(subjects, leave=False):
        descs = os.listdir(f"{mri_path}/{subject}")
        for desc in tqdm(descs, leave=False):
            dates = os.listdir(f"{mri_path}/{subject}/{desc}")
            for date in tqdm(dates, leave=False):
                img_ids = os.listdir(f"{mri_path}/{subject}/{desc}/{date}")
                for img_id in img_ids:
                    img_visit = df_mri.loc[df_mri['Image Data ID'] == img_id]['Visit'].iloc[0]
                    tb_row_idx = df_tb.loc[(df_tb['PTID'] == subject) & (df_tb['VISCODE2'] == img_visit)].index
                    if len(tb_row_idx) != 1:
                        cnt += 1
                        total += 1
                        continue
                    df_tb.loc[tb_row_idx, 'mri_path'] = os.path.join(mri_path, subject, desc, date, img_id)
                    total += 1
    print(f"MRI: {cnt}/{total}")

    subjects = os.listdir(pet_path)
    total = cnt = 0
    for subject in tqdm(subjects, leave=False):
        descs = os.listdir(f"{pet_path}/{subject}")
        for desc in tqdm(descs, leave=False):
            dates = os.listdir(f"{pet_path}/{subject}/{desc}")
            for date in tqdm(dates, leave=False):
                img_ids = os.listdir(f"{pet_path}/{subject}/{desc}/{date}")
                for img_id in img_ids:
                    img_visit = df_pet.loc[df_pet['Image Data ID'] == img_id]['Visit'].iloc[0]
                    tb_row_idx = df_tb.loc[(df_tb['PTID'] == subject) & (df_tb['VISCODE'] == img_visit)].index
                    if len(tb_row_idx) != 1:
                        cnt += 1
                        total += 1
                        continue
                    df_tb.loc[tb_row_idx, 'pet_path'] = os.path.join(pet_path, subject, desc, date, img_id)
                    total += 1
    print(f"PET: {cnt}/{total}")

    df_tb = df_tb.dropna(subset=['mri_path', 'pet_path'])
    df_tb['DIAGNOSIS'] = df_tb['DIAGNOSIS'] - 1

    output_path = 'dataset/tabular/mri_pet_merged.csv'
    df_tb.to_csv(output_path, index=False)
    print(f"Updated DataFrame saved to {output_path}")


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
    plt.show()
    # plt.savefig(f"log/mri/{file_name}.png")
    plt.close(fig)


mri_template = ants.image_read('template/stripped_cropped.nii')


def mri_registration(path):
    moving_image = ants.image_read(path)
    # TODO Add N4
    # corrected_image = ants.n4_bias_field_correction(moving_image, shrink_factor=4, convergence={'iters': [50, 50, 50, 50], 'tol': 1e-7})
    # registration = ants.registration(fixed=mri_template, moving=moving_image, type_of_transform='Rigid')
    # registration = ants.registration(fixed=mri_template, moving=moving_image,type_of_transform='antsRegistrationSyNQuick[s]')
    with tempfile.TemporaryDirectory() as temp_dir:
        outprefix = os.path.join(temp_dir, "registration_")

        registration = ants.registration(
            fixed=mri_template,
            moving=moving_image,
            type_of_transform='Rigid',
            outprefix=outprefix
        )

        return registration["warpedmovout"].numpy()


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = np.maximum(img, 0)  # Clip negatives if any (common in raw MRI)
    mask = img > 0
    foreground = img[mask]
    mean = np.mean(foreground)
    std = np.std(foreground)
    img = (img - mean) / std
    return img.astype(np.float32)


def mri_preprocess(path: str) -> np.ndarray:
    img = mri_registration(path)
    normalized_img = normalize_image(img)
    return normalized_img


def create_mri_dataset():
    df = pd.read_csv('dataset/tabular/img_merged.csv')
    # Define numerical cols for scaling
    numerical_cols = ['MMSCORE', 'TOTSCORE', 'TOTAL13', 'FAQTOTAL', 'PTEDUCAT', 'AGE']

    # First split: 80% train, 20% temp (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df['PTID']))
    train = df.iloc[train_idx]

    # Second split: 50% of temp for val (10% overall), 50% for test (10% overall)
    temp_df = df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df['PTID']))

    # Convert relative indices to absolute
    val_idx = temp_df.iloc[val_idx_rel].index
    test_idx = temp_df.iloc[test_idx_rel].index

    val = df.loc[val_idx]
    test = df.loc[test_idx]

    train = train.copy().reset_index(drop=True)
    val = val.copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)

    scaler = MinMaxScaler()
    scaler.fit(train[numerical_cols])
    train[numerical_cols] = scaler.transform(train[numerical_cols])
    val[numerical_cols] = scaler.transform(val[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])

    os.makedirs('dataset/img/', exist_ok=True)
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        df.to_csv(f'dataset/img/{split_name}.csv', index=False)
        y = df['DIAGNOSIS']

        # Print class distribution
        print(f"\n{split_name.capitalize()} class distribution:")
        print(y.value_counts())

    mri_target = (160, 192, 160)
    with h5py.File('mri_v5.2_Rigid.hdf5', 'w') as h5f:
        ds = {
            'mri_train': h5f.create_dataset('mri_train', (len(train), *mri_target), dtype='float32'),
            'mri_val': h5f.create_dataset('mri_val', (len(val), *mri_target), dtype='float32'),
            'mri_test': h5f.create_dataset('mri_test', (len(test), *mri_target), dtype='float32'),
        }
        for split, df in tqdm([('train', train), ('val', val), ('test', test)]):
            for index, row in tqdm(df.iterrows(), total=len(df)):
                dataset_mri = mri_preprocess(f"{row['image_path']}/brainmask.mgz")
                # log_to_file_image(dataset_mri)
                ds[f'mri_{split}'][index] = dataset_mri


def create_mri_pet_dataset():
    df = pd.read_csv('dataset/tabular/mri_pet_merged.csv')
    # Define numerical cols for scaling
    numerical_cols = ['MMSCORE', 'TOTSCORE', 'TOTAL13', 'FAQTOTAL', 'PTEDUCAT', 'AGE']

    # First split: 80% train, 20% temp (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df['PTID']))
    train = df.iloc[train_idx]

    # Second split: 50% of temp for val (10% overall), 50% for test (10% overall)
    temp_df = df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df['PTID']))

    # Convert relative indices to absolute
    val_idx = temp_df.iloc[val_idx_rel].index
    test_idx = temp_df.iloc[test_idx_rel].index

    val = df.loc[val_idx]
    test = df.loc[test_idx]

    train = train.copy().reset_index(drop=True)
    val = val.copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)

    scaler = MinMaxScaler()
    scaler.fit(train[numerical_cols])
    train[numerical_cols] = scaler.transform(train[numerical_cols])
    val[numerical_cols] = scaler.transform(val[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])

    os.makedirs('dataset/img/', exist_ok=True)
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        df.to_csv(f'dataset/img/{split_name}_mri_pet.csv', index=False)
        y = df['DIAGNOSIS']

        # Print class distribution
        print(f"\n{split_name.capitalize()} class distribution:")
        print(y.value_counts())

    mri_target = (160, 192, 160)
    pet_target = (128, 128, 96)
    with h5py.File('mri_pet_v5.2_Rigid.hdf5', 'w') as h5f:
        ds = {
            'mri_train': h5f.create_dataset('mri_train', (len(train), *mri_target), dtype='float32'),
            'mri_val': h5f.create_dataset('mri_val', (len(val), *mri_target), dtype='float32'),
            'mri_test': h5f.create_dataset('mri_test', (len(test), *mri_target), dtype='float32'),

            'pet_train': h5f.create_dataset('pet_train', (len(train), *pet_target), dtype='uint8'),
            'pet_val': h5f.create_dataset('pet_val', (len(val), *pet_target), dtype='uint8'),
            'pet_test': h5f.create_dataset('pet_test', (len(test), *pet_target), dtype='uint8'),
        }
        for split, df in tqdm([('train', train), ('val', val), ('test', test)]):
            for index, row in tqdm(df.iterrows(), total=len(df)):
                dataset_mri = mri_preprocess(f"{row['mri_path']}/brainmask.mgz")
                pet_image = read_image(row['pet_path'])
                dataset_pet = pet_preprocess2(pet_image)
                ds[f'mri_{split}'][index] = dataset_mri
                ds[f'pet_{split}'][index] = dataset_pet


def recreate_mri_dataset():
    df = pd.read_csv('dataset/tabular/img_merged.csv')
    df = df[df['DIAGNOSIS'] != 1]
    df['DIAGNOSIS'] = df['DIAGNOSIS'] / 2
    # Define numerical cols for scaling
    numerical_cols = ['MMSCORE', 'TOTSCORE', 'TOTAL13', 'FAQTOTAL', 'PTEDUCAT', 'AGE']

    # First split: 80% train, 20% temp (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df['PTID']))
    train = df.iloc[train_idx]

    # Second split: 50% of temp for val (10% overall), 50% for test (10% overall)
    temp_df = df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df['PTID']))

    # Convert relative indices to absolute
    val_idx = temp_df.iloc[val_idx_rel].index
    test_idx = temp_df.iloc[test_idx_rel].index

    val = df.loc[val_idx]
    test = df.loc[test_idx]

    train = train.copy().reset_index(drop=True)
    val = val.copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)

    scaler = MinMaxScaler()
    scaler.fit(train[numerical_cols])
    train[numerical_cols] = scaler.transform(train[numerical_cols])
    val[numerical_cols] = scaler.transform(val[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])

    os.makedirs('dataset/img/', exist_ok=True)
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        df.to_csv(f'dataset/img/2c_{split_name}.csv', index=False)
        y = df['DIAGNOSIS']

        # Print class distribution
        print(f"\n{split_name.capitalize()} class distribution:")
        print(y.value_counts())

    mri_target = (160, 192, 160)
    old_train_df = pd.read_csv('dataset/img/train.csv')
    old_val_df = pd.read_csv('dataset/img/val.csv')
    old_test_df = pd.read_csv('dataset/img/test.csv')
    # Open old HDF5 once for reading
    with h5py.File('dataset/mri_v5.2_Rigid.hdf5', 'r') as old_h5f:
        with h5py.File('2c_mri_v5.2_Rigid.hdf5', 'w') as new_h5f:
            ds = {
                'mri_train': new_h5f.create_dataset('mri_train', (len(train), *mri_target), dtype='float32'),
                'mri_val': new_h5f.create_dataset('mri_val', (len(val), *mri_target), dtype='float32'),
                'mri_test': new_h5f.create_dataset('mri_test', (len(test), *mri_target), dtype='float32'),
            }
            for split, split_df in tqdm([('train', train), ('val', val), ('test', test)], desc="Processing splits"):
                for index, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split} rows"):
                    image_path = row['image_path']
                    match_train = old_train_df[old_train_df['image_path'] == image_path]
                    match_val = old_val_df[old_val_df['image_path'] == image_path]
                    match_test = old_test_df[old_test_df['image_path'] == image_path]
                    if len(match_train) > 0:
                        dataset_mri = old_h5f[f'mri_train'][match_train.index[0]]
                    elif len(match_val) > 0:
                        dataset_mri = old_h5f[f'mri_val'][match_val.index[0]]
                    elif len(match_test) > 0:
                        dataset_mri = old_h5f[f'mri_test'][match_test.index[0]]
                    else:
                        print(
                            f"Warning: No unique match found for image_path '{image_path}' in old datasets. Skipping.")
                        continue
                    ds[f'mri_{split}'][index] = dataset_mri


def create_mci_dataset():
    df = pd.read_csv('dataset/tabular/img_merged.csv')
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'], format='%Y-%m-%d')
    mci = df[df['DIAGNOSIS'] == 1].copy().reset_index(drop=True)
    df = df[df['DIAGNOSIS'] != 0]
    stable = progress = no_data = 0
    for index, row in mci.iterrows():
        history = df[df['PTID'] == row['PTID']].sort_values(by='EXAMDATE')
        history['delta_days'] = (history['EXAMDATE'] - row['EXAMDATE']).dt.days
        filtered_history = history[(history['delta_days'] > 0) & (history['delta_days'] <= 730)].copy()
        if filtered_history.empty:
            no_data += 1
            mci.loc[index, 'DIAGNOSIS'] = -1
            continue
        else:
            mci.loc[index, 'DIAGNOSIS'] = filtered_history.iloc[-1]['DIAGNOSIS'] - 1
            if filtered_history.iloc[0]['DIAGNOSIS'] == 1:
                stable += 1
            elif filtered_history.iloc[0]['DIAGNOSIS'] == 2:
                progress += 1
            else:
                no_data += 1

    mci = mci[mci['DIAGNOSIS'] != -1].reset_index(drop=True)

    # ad_rows = history[history['DIAGNOSIS'] == 2]
    # if ad_rows.empty:
    #     no_data += 1
    #     mci.loc[index, 'DIAGNOSIS'] = -1
    #     continue
    # first_ad_date = ad_rows['EXAMDATE'].min()
    # delta = (first_ad_date - row['EXAMDATE']).days
    #
    # if delta <= 730:
    #     progress += 1
    #     # mci.iloc[index]['DIAGNOSIS'] = 1
    # else:
    #     stable += 1
    #     mci.loc[index, 'DIAGNOSIS'] = 0

    # if 2 in history['DIAGNOSIS'].values:
    #     progress += 1
    # else:
    #     stable += 1

    print(progress, stable, no_data)
    df = mci

    # df['DIAGNOSIS'] = df['DIAGNOSIS'] / 2

    # exit(0)
    # Define numerical cols for scaling
    numerical_cols = ['MMSCORE', 'TOTSCORE', 'TOTAL13', 'FAQTOTAL', 'PTEDUCAT', 'AGE']

    # First split: 80% train, 20% temp (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss1.split(df, groups=df['PTID']))
    train = df.iloc[train_idx]

    # Second split: 50% of temp for val (10% overall), 50% for test (10% overall)
    temp_df = df.iloc[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_rel, test_idx_rel = next(gss2.split(temp_df, groups=temp_df['PTID']))

    # Convert relative indices to absolute
    val_idx = temp_df.iloc[val_idx_rel].index
    test_idx = temp_df.iloc[test_idx_rel].index

    val = df.loc[val_idx]
    test = df.loc[test_idx]

    train = train.copy().reset_index(drop=True)
    val = val.copy().reset_index(drop=True)
    test = test.copy().reset_index(drop=True)

    scaler = MinMaxScaler()
    scaler.fit(train[numerical_cols])
    train[numerical_cols] = scaler.transform(train[numerical_cols])
    val[numerical_cols] = scaler.transform(val[numerical_cols])
    test[numerical_cols] = scaler.transform(test[numerical_cols])

    os.makedirs('dataset/img/', exist_ok=True)
    for split_name, df in [('train', train), ('val', val), ('test', test)]:
        df.to_csv(f'dataset/img/mci_{split_name}.csv', index=False)
        y = df['DIAGNOSIS']

        # Print class distribution
        print(f"\n{split_name.capitalize()} class distribution:")
        print(y.value_counts())

    mri_target = (160, 192, 160)
    old_train_df = pd.read_csv('dataset/img/train.csv')
    old_val_df = pd.read_csv('dataset/img/val.csv')
    old_test_df = pd.read_csv('dataset/img/test.csv')
    # Open old HDF5 once for reading
    with h5py.File('dataset/mri_v5.2_Rigid.hdf5', 'r') as old_h5f:
        with h5py.File('mci-2y_v5.2_Rigid.hdf5', 'w') as new_h5f:
            ds = {
                'mri_train': new_h5f.create_dataset('mri_train', (len(train), *mri_target), dtype='float32'),
                'mri_val': new_h5f.create_dataset('mri_val', (len(val), *mri_target), dtype='float32'),
                'mri_test': new_h5f.create_dataset('mri_test', (len(test), *mri_target), dtype='float32'),
            }
            for split, split_df in tqdm([('train', train), ('val', val), ('test', test)], desc="Processing splits"):
                for index, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split} rows"):
                    image_path = row['image_path']
                    match_train = old_train_df[old_train_df['image_path'] == image_path]
                    match_val = old_val_df[old_val_df['image_path'] == image_path]
                    match_test = old_test_df[old_test_df['image_path'] == image_path]
                    if len(match_train) > 0:
                        dataset_mri = old_h5f[f'mri_train'][match_train.index[0]]
                    elif len(match_val) > 0:
                        dataset_mri = old_h5f[f'mri_val'][match_val.index[0]]
                    elif len(match_test) > 0:
                        dataset_mri = old_h5f[f'mri_test'][match_test.index[0]]
                    else:
                        print(
                            f"Warning: No unique match found for image_path '{image_path}' in old datasets. Skipping.")
                        continue
                    ds[f'mri_{split}'][index] = dataset_mri


def run():
    # merge_mri_csv('dataset/MRI2/ADNI/')
    # merge_mri_csv_pet('dataset/MRI2/ADNI/', 'dataset/PET/ADNI/')
    # create_mri_dataset()
    create_mri_pet_dataset()
    # recreate_mri_dataset()
    # create_mci_dataset()
