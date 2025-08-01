import os

import ants
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


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
    registration = ants.registration(fixed=mri_template, moving=moving_image, type_of_transform='SyN')
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
    with h5py.File('mri_v5.2_SyN.hdf5', 'w') as h5f:
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


def run():
    merge_mri_csv('dataset/MRI2/ADNI/')
    # create_mri_dataset()
