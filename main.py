# from seg.patch import print_patch_sizes, extract_patches, get_patch_indices
#
# get_patch_indices()

# from train.vq_gan_mri import run
#
# run()

# from train.pasta import run
#
# run()

# from train.cvit import run
#
# run()

from dataset import mri_data_paths, create_mri_dataset

create_mri_dataset(mri_data_paths)
