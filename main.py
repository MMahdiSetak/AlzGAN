# from seg.patch import print_patch_sizes, extract_patches, get_patch_indices
#
# get_patch_indices()
import torch
from train.classifier import run

torch.set_float32_matmul_precision('medium')
run()

# from dataset import create_pet_dataset, pet_data_path
#
# create_pet_dataset(pet_data_path)

# import optuna
#
# from train.opt_hparm import objective
#
# study = optuna.create_study(study_name='cvit', direction="maximize", storage="sqlite:///cvit_study.db",
#                             load_if_exists=True)
# study.optimize(objective, n_trials=50, gc_after_trial=True)
#
# best_params = study.best_params
# print(f"Best Hyperparameters: {best_params}")
