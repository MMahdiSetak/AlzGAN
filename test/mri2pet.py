import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchmetrics.image import PeakSignalNoiseRatio

from model.MRI2PET.model import MRI2PET
from model.dataloader import MRI2PETDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "log/mri2pet/version_21/checkpoints/mri2pet_best_model.ckpt"
datapath = 'dataset/mri_pet_label_v5.hdf5'
batch_size = 2
num_workers = 4
test_loader = DataLoader(
    dataset=MRI2PETDataset(datapath, 'train'),
    batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False
)
lit_model = MRI2PET.load_from_checkpoint(checkpoint_path=checkpoint_path)


def run():
    # ----- INFERENCE -----
    psnr = PeakSignalNoiseRatio(data_range=1).to(device)
    # metrics = {
    #     "PSNR": PeakSignalNoiseRatio(data_range=1),
    # }
    # train_metrics = MetricCollection(metrics, postfix="/train")
    mris = []
    pets = []
    gen_pets = []
    lit_model.to(device)
    lit_model.eval()
    lit_model.freeze()
    with torch.no_grad():
        for batch in test_loader:
            mri, pet, _ = batch
            pet = pet.to(device)
            mri = mri.to(device)
            # log_video(mri[0].squeeze().cpu().numpy())
            gen_pet = lit_model.model.generate(mri)
            print(psnr(pet, gen_pet))
            continue

            pets.append(pet[:, :, :, :, 32].cpu())
            mris.append(mri[:, :, :, :, 32].cpu())
            gen_pets.append(gen_pet[:, :, :, :, 32].cpu())
            break

    pets = torch.cat(pets, dim=0)
    mris = torch.cat(mris, dim=0)
    gen_pets = torch.cat(gen_pets, dim=0)
    for i in range(len(mris)):
        vutils.save_image(pets[i], f"test/{i:03d}_pet.png", normalize=True)
        vutils.save_image(mris[i], f"test/{i:03d}_mri.png", normalize=True)
        vutils.save_image(gen_pets[i], f"test/{i:03d}_gen_pet.png", normalize=True)
        # log_video(pets[i], f"test/{i:03d}_gen_pet")
        # log_video(mri[i], f"test/{i:03d}_gen_pet")
        # log_video(gen_pets[i], f"test/{i:03d}_gen_pet")
