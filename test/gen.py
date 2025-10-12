from data.image import pair_log
from model.dataloader import LcDDPMDataset
from model.log import log_3d
from model.vq_vae_3d.vqvae import VQVAE


def run():
    # datapath = 'dataset/mri_pet_v5.2_Rigid.hdf5'
    datapath = 'dataset/mri_pet_label_v5.1_Rigid.hdf5'
    index = 12
    dataset = LcDDPMDataset(datapath, 'train')
    mri = dataset[index][0].squeeze()
    pet = dataset[index][1].squeeze()

    # pair_log(mri=mri, pet=pet, filename='chert')

    mri_vae_model = VQVAE.load_from_checkpoint(
        checkpoint_path="../final models/mri autoencoder/version_38/checkpoints/latest_checkpoint.ckpt")
    mri_vae_model.eval()
    pet_vae_model = VQVAE.load_from_checkpoint(
        checkpoint_path="../final models/pet autoencoder/version_0/checkpoints/latest_checkpoint.ckpt")
    pet_vae_model.eval()
    # ddpm_model = LcDDPM.load_from_checkpoint(checkpoint_path=ddpm_checkpoint)

    _, fake_mri, _ = mri_vae_model(mri.unsqueeze(0).unsqueeze(0).to('cuda'))
    log_3d(fake_mri.detach().to('cpu').numpy().squeeze())
    _, fake_pet, _ = pet_vae_model(pet.unsqueeze(0).unsqueeze(0).to('cuda'))

    print("Done!")
