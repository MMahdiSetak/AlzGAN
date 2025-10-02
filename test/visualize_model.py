import hydra
import imageio
import numpy as np
import torch
from captum.attr import Occlusion
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import captum.attr as attr

from model.classifier.model import Classifier
from model.dataloader import MergedDataset
from model.log import log_3d


@hydra.main(config_path='../config/model', config_name='classifier', version_base=None)
def run(cfg: DictConfig):
    train_dataset = MergedDataset(csv_path=cfg.tabular_dataset, hdf5_path=cfg.mri_dataset, split='train', mri=True,
                                  pet=False,
                                  mri_cache=None, apply_augmentation=False)

    model = Classifier.load_from_checkpoint(
        checkpoint_path='log/classifier/version_75/checkpoints/classifier_best_model.ckpt')
    model.eval()  # Temp eval mode for stable predictions

    sample = train_dataset[1]
    mri = sample['mri'].to(device='cuda').unsqueeze(0)
    tabular = sample['tabular'].to(device='cuda').unsqueeze(0)
    with torch.no_grad():
        # outputs = model(mri, tabular)
        outputs = model(mri)
    pred_class = outputs.argmax(dim=1).item()

    # todo try different algorithms
    saliency = attr.Saliency(model)
    # saliency = attr.IntegratedGradients(model)
    # occlusion = Occlusion(model)
    # Compute gradients w.r.t. inputs for the target class
    # attributions = saliency.attribute(inputs=(mri, tabular), target=pred_class)
    attributions = saliency.attribute(inputs=mri, target=pred_class)
    # attributions = saliency.attribute(inputs=(mri, tabular),
    #                                   baselines=(torch.ones_like(mri) * mri.min(), torch.zeros_like(tabular)),
    #                                   target=pred_class,
    #                                   internal_batch_size=2)

    # attributions = occlusion.attribute(inputs=(mri, tabular),
    #                                    strides=((1, 16, 16, 16), (13,)),
    #                                    target=pred_class,
    #                                    sliding_window_shapes=((1, 16, 16, 16), (13,)),
    #                                    baselines=0)
    mri_attr = attributions[0]  # Only MRI attributions (ignore tabular)
    mri_attr = mri_attr.squeeze().squeeze().detach().cpu().numpy()
    original_mri = mri.squeeze().squeeze().detach().cpu().numpy()
    # log_3d(mri_attr.squeeze().squeeze().detach().cpu().numpy(), title=f'mri_attr')

    # mri_attr = np.abs(mri_attr)
    mri_attr = (mri_attr - mri_attr.min()) / (mri_attr.max() - mri_attr.min())
    # original_mri = (original_mri - original_mri.min()) / (original_mri.max() - original_mri.min())
    slice_idx = mri.shape[2] // 2  # Middle slice
    original_slice = original_mri[:, :, slice_idx]  # [H, W]
    attr_slice = mri_attr[:, :, slice_idx]  # [H, W]

    # Visualize and save 2D slice
    plt.figure(figsize=(10, 5))
    # Original MRI slice
    plt.subplot(1, 2, 1)
    plt.imshow(original_slice, cmap='gray')
    plt.title('Original MRI Slice')
    plt.axis('off')
    # Attribution heatmap overlaid
    plt.subplot(1, 2, 2)
    plt.imshow(original_slice, cmap='gray')
    plt.imshow(attr_slice, cmap='inferno', alpha=0.7)  # Overlay with transparency
    plt.title(f'Saliency Map (Class {pred_class})')
    plt.axis('off')
    plt.show()
    # plt.savefig('mri_attribution.png')
    plt.close()

    H, W, Z = original_mri.shape
    name = 'saliency_test'
    writer = imageio.get_writer(f'{name}.mp4', fps=8, codec='libx264')
    for z in range(Z):
        original_slice = original_mri[:, :, z]
        attr_slice = mri_attr[:, :, z]

        # Create figure for overlay
        fig = plt.figure(figsize=(5.12, 5.12))
        plt.imshow(original_slice, cmap='gray')
        plt.imshow(attr_slice, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
        plt.title(f'Saliency Map (Class {pred_class}, Slice {z})')
        plt.axis('off')

        # Convert figure to NumPy array
        fig.canvas.draw()
        # Get RGB buffer and convert to NumPy
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # [height, width, 3]

        # Append to video
        writer.append_data(img_array)
        plt.close()

    writer.close()

    print("good")


@hydra.main(config_path='../config/model', config_name='classifier', version_base=None)
def run2(cfg: DictConfig):
    train_dataset = MergedDataset(csv_path=cfg.tabular_dataset, hdf5_path=cfg.mri_dataset, split='train')

    model = Classifier.load_from_checkpoint(
        checkpoint_path='log/classifier/version_48/checkpoints/classifier_best_model.ckpt')
    model.eval()  # Temp eval mode for stable predictions

    sample = train_dataset[1]
    mri = sample['mri'].to(device='cuda').unsqueeze(0)
    tabular = sample['tabular'].to(device='cuda').unsqueeze(0)
    with torch.no_grad():
        outputs = model(mri, tabular)
    pred_class = outputs.argmax(dim=1).item()

    saliency = attr.IntegratedGradients(model)
    attributions = saliency.attribute(inputs=(mri, tabular),
                                      baselines=(torch.ones_like(mri) * mri.min(), torch.zeros_like(tabular)),
                                      target=pred_class,
                                      internal_batch_size=16)
    # attributions = saliency.attribute(inputs=(mri, tabular), target=pred_class)
    mri_attr = attributions[0]  # Only MRI attributions (ignore tabular)
    mri_attr = mri_attr.squeeze().squeeze().detach().cpu().numpy()
    original_mri = mri.squeeze().squeeze().detach().cpu().numpy()
    # log_3d(mri_attr.squeeze().squeeze().detach().cpu().numpy(), title=f'mri_attr')

    mri_attr = np.abs(mri_attr)
    # mri_attr = np.clip(mri_attr, 0, None)
    mri_attr = (mri_attr - mri_attr.min()) / (mri_attr.max() - mri_attr.min())

    # mean = np.mean(mri_attr)
    # std = np.std(mri_attr)
    # mri_attr = (mri_attr - mean) / std
    # original_mri = (original_mri - original_mri.min()) / (original_mri.max() - original_mri.min())
    slice_idx = mri.shape[2] // 2  # Middle slice
    original_slice = original_mri[:, :, slice_idx]  # [H, W]
    attr_slice = mri_attr[:, :, slice_idx]  # [H, W]

    # Visualize and save 2D slice
    plt.figure(figsize=(10, 5))
    # Original MRI slice
    plt.subplot(1, 2, 1)
    plt.imshow(original_slice, cmap='gray')
    plt.title('Original MRI Slice')
    plt.axis('off')
    # Attribution heatmap overlaid
    plt.subplot(1, 2, 2)
    plt.imshow(original_slice, cmap='gray')
    plt.imshow(attr_slice, cmap='viridis', alpha=0.7)  # Overlay with transparency
    plt.title(f'Saliency Map (Class {pred_class})')
    plt.axis('off')
    plt.show()
    # plt.savefig('mri_attribution.png')
    plt.close()

    H, W, Z = original_mri.shape
    name = 'saliency_IntegratedGradients'
    writer = imageio.get_writer(f'{name}.mp4', fps=8, codec='libx264')
    for z in range(Z):
        original_slice = original_mri[:, :, z]
        attr_slice = mri_attr[:, :, z]

        # Create figure for overlay
        fig = plt.figure(figsize=(5.12, 5.12))
        plt.imshow(original_slice, cmap='gray')
        plt.imshow(attr_slice, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        plt.title(f'Saliency Map (Class {pred_class}, Slice {z})')
        plt.axis('off')

        # Convert figure to NumPy array
        fig.canvas.draw()
        # Get RGB buffer and convert to NumPy
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # [height, width, 3]

        # Append to video
        writer.append_data(img_array)
        plt.close()

    writer.close()

    print("better!")
