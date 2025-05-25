import torch
from pytorch_lightning import Trainer
from model.ddpm.diffusion import Diffusion
from model.ddpm.trainer import GaussianDiffusion
from model.ddpm.unet import create_model
from model.dataloader import DDPMPairDataset
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# ----- CONFIGURATION -----
checkpoint_path = "log/ddpm/version_47/checkpoints/ddpm_best_model.ckpt"
datapath = 'dataset/mri_pet_label_v4.hdf5'

# Match these with your config or training script
input_size = 128
depth_size = 128
num_channels = 128
num_res_blocks = 1
timesteps = 250
in_channels = 2
out_channels = 1
batch_size = 2
num_workers = 8

# ----- MODEL SETUP -----
model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels, out_channels=out_channels)
diffusion = GaussianDiffusion(
    model,
    image_size=input_size,
    depth_size=depth_size,
    timesteps=timesteps,  # number of steps
    loss_type='l1',  # L1 or L2
    channels=out_channels
)
lit_model = Diffusion.load_from_checkpoint(checkpoint_path)

# ----- DATA LOADING -----
test_dataset = DDPMPairDataset(datapath, 'test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)


def run():
    # ----- INFERENCE -----
    mris = []
    pets = []
    gen_pets = []
    lit_model.eval()
    lit_model.freeze()
    with torch.no_grad():
        for batch in test_loader:
            mri, pet, _ = batch
            bs = pet.size(0)
            get_pet = lit_model.model.sample(bs, mri)
            pets.append(pet[:, :, :, :, 64].cpu())
            mris.append(mri[:, :, :, :, 64].cpu())
            gen_pets.append(get_pet[:, :, :, :, 64].cpu())
            break

    pets = torch.cat(pets, dim=0)
    mris = torch.cat(mris, dim=0)
    gen_pets = torch.cat(gen_pets, dim=0)
    for i in range(len(mris)):
        vutils.save_image(pets[i], f"{i:03d}_pet.png", normalize=True)
        vutils.save_image(mris[i], f"{i:03d}_mri.png", normalize=True)
        vutils.save_image(gen_pets[i], f"{i:03d}_gen_pet.png", normalize=True)
