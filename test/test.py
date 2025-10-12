
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from model.classifier.model import Classifier
from model.dataloader import MergedDataset



@hydra.main(config_path='../config/model', config_name='classifier', version_base=None)
def run(cfg: DictConfig):
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers
    mri_dataset = cfg.mri_dataset
    mri = cfg.mri_data
    pet = cfg.real_pet

    trainer = pl.Trainer(
        max_epochs=cfg.max_epoch,
        strategy=DDPStrategy(find_unused_parameters=True),
        num_sanity_val_steps=0,
        accelerator="auto",
        # strategy="fsdp",
        # devices=[0, 1],
        # overfit_batches=3,
        val_check_interval=1.0,
        # logger=logger,
        gradient_clip_val=1.0,
        precision='16-mixed',
        # callbacks=[early_stop_callback, checkpoint_callback],
        # accumulate_grad_batches=2,
        log_every_n_steps=5,
        enable_checkpointing=True,
    )


    test_loader = DataLoader(
        # dataset=MRIDataset(data_path=datapath, split='test'),
        # dataset=MRIDataset(*test_ram_loader.get_data(), split='test'),
        dataset=MergedDataset(csv_path=cfg.tabular_dataset, hdf5_path=mri_dataset, mri=mri, pet=pet, split='test'),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )
    # model_path = os.path.join(logger.log_dir, "checkpoints", "classifier_best_model.ckpt")
    model_path = "/home/m-mahdi/Desktop/uni/thesis/final models/classifier/version_76/checkpoints/classifier_best_model.ckpt"
    model = Classifier.load_from_checkpoint(model_path)
    trainer.test(model=model, dataloaders=test_loader)