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

# from dataset import mri_data_paths, create_mri_pet_label_dataset, pet_data_path
#
# create_mri_pet_label_dataset(mri_data_paths, pet_data_path)

import optuna

from train.opt_hparm import objective

# Create the Optuna study and run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Get the best hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")
