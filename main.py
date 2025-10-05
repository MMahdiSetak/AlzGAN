# from seg.patch import print_patch_sizes, extract_patches, get_patch_indices
#
# get_patch_indices()

if __name__ == '__main__':
    import torch
    from data.image_tabular import run

    torch.set_float32_matmul_precision('medium')
    run()

    # from data.image import create_mri_dataset, mri_data_path, create_mri_pet_label_dataset, pet_data_path

    # create_mri_dataset(mri_path=mri_data_path)
    # create_mri_pet_label_dataset(mri_path=mri_data_path, pet_path=pet_data_path)
    print("Done! 🎉🎊")
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
