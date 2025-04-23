from model.ddpm.diffusion_model.trainer import GaussianDiffusion, Trainer
from model.ddpm.diffusion_model.unet import create_model
from model.dataloader import DDPMPairDataset

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

input_size = 128
depth_size = 128
num_channels = 128
num_res_blocks = 1
save_and_sample_every = 1000
with_condition = True
train_lr = 1e-5
timesteps = 250
batchsize = 1
epochs = 50000

in_channels = 2
out_channels = 1


def run():
    model = create_model(input_size, num_channels, num_res_blocks, in_channels=in_channels,
                         out_channels=out_channels).to(device)

    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=timesteps,  # number of steps
        loss_type='l1',  # L1 or L2
        with_condition=with_condition,
        channels=out_channels
    ).to(device)

    dataset = DDPMPairDataset('dataset/mri_pet_label_v3.hdf5', 'train')

    trainer = Trainer(
        diffusion,
        dataset,
        image_size=input_size,
        depth_size=depth_size,
        train_batch_size=batchsize,
        train_lr=train_lr,
        train_num_steps=epochs,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        fp16=False,  # True,                       # turn on mixed precision training with apex
        with_condition=with_condition,
        save_and_sample_every=save_and_sample_every,
    )

    trainer.train()
