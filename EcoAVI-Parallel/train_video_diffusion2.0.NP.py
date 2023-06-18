
import torch
from video_diffusion_pytorchAVI2 import Unet3DNP, GaussianDiffusionNP, TrainerNP
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary


    
os.environ['CUDA_VISIBLE_DEVICES']='6, 5'
device='cuda:0'

script_path = os.path.dirname(__file__)

device0 = 'cuda:0'
device1 = 'cuda:1'
device2 = 'cuda:2'

unet = Unet3DNP(
    dim = 64,
    device0 = device0,  #needs to be the same as the device from GaussianDiffusion and the trainer
    device1 = device1,
    device2 = device2
)

diffusion = GaussianDiffusionNP(
    height = 96,
    width = 128,
    num_frames = 16,
    timesteps = 1000,   # number of steps
    device = device0    #device where the alphas will be stored. needs to be the same as where the training data is stored at the beginning
)

trainer = TrainerNP(
    unet,
    diffusion,
    os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'newpreprocessedData', 'joinedframes_128x96')),                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 4,
    train_lr = 1e-4,
    save_and_sample_every = 1000,
    results_folder = os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'cos_tau_3')),
    train_num_steps = 126000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    num_sample_rows = 1,
    loss_type = 'l1',    # L1 or L2
    sample = False,
    continuein = 0
)



trainer.train(dev=device0) #device where the training data will be stored at the beginning
