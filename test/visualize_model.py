import hydra
import torch
from omegaconf import DictConfig
import captum.attr as attr

from model.classifier.model import Classifier
from model.dataloader import MergedDataset


@hydra.main(config_path='../config/model', config_name='classifier', version_base=None)
def run(cfg: DictConfig):
    train_dataset = MergedDataset(csv_path=cfg.tabular_dataset, hdf5_path=cfg.mri_dataset, split='train')

    model = Classifier.load_from_checkpoint(
        checkpoint_path='log/classifier/version_48/checkpoints/classifier_best_model.ckpt')

    sample = train_dataset[1]
    mri = sample['mri']
    tabular = sample['tabular']

    with torch.no_grad():
        outputs = model(mri, tabular)
    pred_class = outputs.argmax(dim=1).item()

    model.eval()  # Temp eval mode for stable predictions
    saliency = attr.Saliency(model)
    # Compute gradients w.r.t. inputs for the target class
    attributions = saliency.attribute(inputs=(mri, tabular), target=pred_class)
    mri_attr = attributions[0]  # Only MRI attributions (ignore tabular)

    # Visualize and log
    # self.visualize_and_log_attribution(mri_sample, mri_attr, self.current_epoch)

    model.eval()
