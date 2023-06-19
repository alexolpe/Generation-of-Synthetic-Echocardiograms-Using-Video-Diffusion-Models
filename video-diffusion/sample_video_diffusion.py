
import torch
from video_diffusion import Unet3D, GaussianDiffusion, Trainer
import os
import math
import copy
import torch
import torchvision
import cv2
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from video_diffusion_pytorch.text import tokenize, bert_embed, BERT_MODEL_DIM
from prettytable import PrettyTable
from torchsummary import summary

import shutil

os.environ['CUDA_VISIBLE_DEVICES']='3, 4'
device='cuda:1'

script_path = os.path.dirname(__file__)

device0 = 'cuda:0'
device1 = 'cuda:1'
device2 = 'cuda:2'

unet = Unet3D(
    dim = 64,
    device0 = device0,  #needs to be the same as the device from GaussianDiffusion and the trainer
    device1 = device1,
    device2 = device2
)

diffusion = GaussianDiffusion(
    height = 96,
    width = 128,
    num_frames = 16,
    timesteps = 1000,   # number of steps
    device = device0    #device where the alphas will be stored. needs to be the same as where the training data is stored at the beginning
)

'''def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    array = tensor.detach().cpu().numpy()

    # reshape the numpy array to video dimensions
    array = np.transpose(array, (1, 2, 3, 0))
    array = np.uint8(array * 255)

    # create video using OpenCV
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (tensor.shape[3], tensor.shape[2]), False)
    for i in range(tensor.shape[1]):
        out.write(array[i])
    out.release()
    return '''

'''def video_to_images(video_tensor, output_folder, num_video):
    num_canales, num_frames, altura, anchura = video_tensor.size()
    final_folder = os.path.join(output_folder, num_video)
    os.makedirs(final_folder, exist_ok=True)
    # Itera sobre cada frame y guárdalo como una imagen TIFF en escala de grises
    for i in range(num_frames):
        # Obtén el frame actual
        frame = video_tensor[:, i, :, :]
        print('frame', frame.shape)
        
        normalized_frame = (frame - frame.min()) / (frame.max() - frame.min())
        normalized_frame = (normalized_frame * 255).byte()
    
        normalized_frame = normalized_frame.squeeze()
        print('frame', normalized_frame.shape)
        # Crea una nueva imagen PIL a partir del tensor
        imagen = Image.fromarray(normalized_frame.cpu().byte().numpy(), mode='L')
        image_path = f"{final_folder}/frame_{i}_116.tiff"
        # Guarda la imagen en formato TIFF
        imagen.save(image_path)'''
        
def video_to_images(video_tensor, output_folder, num_video):
    num_canales, num_frames, altura, anchura = video_tensor.size()
    # Itera sobre cada frame y guárdalo como una imagen PNG en escala de grises
    for i in range(num_frames):
        # Obtén el frame actual
        frame = video_tensor[:, i, :, :]
        print('frame', frame.shape)
        
        normalized_frame = (frame - frame.min()) / (frame.max() - frame.min())
        normalized_frame = (normalized_frame * 255).byte()
    
        normalized_frame = normalized_frame.squeeze()
        print('frame', normalized_frame.shape)
        # Crea una nueva imagen PIL a partir del tensor
        imagen = Image.fromarray(normalized_frame.cpu().byte().numpy(), mode='L')
        image_path = f"{output_folder}/{num_video}-frame_{i}_125.png"
        # Guarda la imagen en formato PNG
        imagen.save(image_path)

unet.load_state_dict(torch.load(os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'cos_tau_3', f'model-{125}.pt'))))
video_path_tiff_fvd = "/data/aolivepe/cos_tau_3_sample"

for n in reversed(range(224)):
    #diffusion.sample(unet = unet, batch_size = 1)

    num_sample_rows = 1

    num_samples = num_sample_rows ** 2 #A: num_samples=16
    #batches = num_to_groups(num_samples, batch_size)#A: batches = [4,4,4,4]
    batches = [1]

    all_videos_list = list(map(lambda n: diffusion.sample(unet = unet, batch_size=n), batches))
    all_videos_list = torch.cat(all_videos_list, dim = 0)

    one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = num_sample_rows)
    video_to_images(one_gif, video_path_tiff_fvd, str(n))