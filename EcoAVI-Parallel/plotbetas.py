import math
import copy
import torch
import torchvision
import cv2
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
import os

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

import torch
import torchvision
from torchvision import transforms
from PIL import Image

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

def cosine_beta_schedule(timesteps, tau, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** (2*tau)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep = 1000, linear_start=1e-4, linear_end= 0.9999, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 0.9999, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

def identity(t, *args, **kwargs):
    return t

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    array = tensor.detach().cpu().numpy()

    # reshape the numpy array to video dimensions
    array = np.transpose(array, (1, 2, 3, 0))
    array = np.uint8(array * 255)

    # create video using OpenCV
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (tensor.shape[3], tensor.shape[2]), False)
    for i in range(tensor.shape[1]):
        out.write(array[i])
    out.release()
    return 

def normalize_img(t):
    return t * 2 - 1

def extract(a, t, x_shape):
    b, *_ = t.shape #A: el "*_" serveix per trencar l'estructura torch.Size([size]) amb la que et retorna pytorch la mida i quedarte nomes amb el valor size
    print('a', a.shape)
    print('t', t.shape)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def q_sample(x, t):
    b = x.shape[0]
    x = normalize_img(x)
    real_noise = torch.randn_like(x)
        
    betas1 = cosine_beta_schedule(1000, 1)
    betas2 = cosine_beta_schedule(1000, 2)
    betas3 = cosine_beta_schedule(1000, 3)
    print('betas', betas1.shape)
    alphas1 = 1. - betas1
    alphas2 = 1. - betas2
    alphas3 = 1. - betas3
    snr1 = alphas1/betas1
    snr2 = alphas2/betas2
    snr3 = alphas3/betas3
    alphas_cumprod = torch.cumprod(alphas1, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    x_noisy = extract(sqrt_alphas_cumprod, t, x.shape) * x + extract(sqrt_one_minus_alphas_cumprod, t, x.shape) * real_noise
    return x_noisy, t, real_noise, alphas1, alphas2, alphas3

def video_to_tiff(video_tensor, output_folder, num_video, t):
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
        imagen = Image.fromarray(normalized_frame.byte().numpy(), mode='L')
        image_path = f"{final_folder}/frame_{i}_timestep{t}.tiff"
        # Guarda la imagen en formato TIFF
        imagen.save(image_path)
        
def video_to_png(video_tensor, output_folder, num_video, t):
    num_channels, num_frames, height, width = video_tensor.size()
    final_folder = os.path.join(output_folder, num_video)
    os.makedirs(final_folder, exist_ok=True)
    transform = transforms.Grayscale()
    for frame_idx in range(num_frames):
        frame_tensor = video_tensor[:, frame_idx, :, :]
        pil_image = transforms.ToPILImage()(frame_tensor)
        image_path = os.path.join(output_folder, f"frame_{frame_idx}_timestep{t}.png")
        pil_image.save(image_path)
    

totensor = torchvision.transforms.ToTensor()
transform = T.Compose([
            #T.Resize(image_size),
            T.Lambda(identity),
            #T.CenterCrop(image_size),
            #T.ToTensor()
        ])
cast_num_frames_fn = identity

#TIFF
path = '/data/aolivepe/tiff'
video = []
for nombre_archivo in sorted(os.listdir(path)):
    if nombre_archivo.endswith('.tiff'):
        ruta_completa = os.path.join(path, nombre_archivo)
        imagen = cv2.imread(ruta_completa)
        gray_image = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        frame_tensor = totensor(gray_image)
        video.append(frame_tensor)

video_tensor = torch.stack(video, dim=0)
video_tensor=torch.transpose(video_tensor, 0,1)


'''#PNG
path = '/data/aolivepe/newpreprocessedData/videos_128x96_folder_frames/1-video001_38x384x512(0)'
video = []
for filename in os.listdir(path):
    if filename.endswith('.png'):
        file_path = os.path.join(path, filename)
        frame = Image.open(file_path).convert("L")
        frame_tensor = totensor(frame)
        frame_tensor = transform(frame_tensor)
        video.append(frame_tensor)
        
video_tensor = torch.stack(video, dim=0)
video_tensor=torch.transpose(video_tensor, 0,1)'''


#AVI
'''path = '/data/aolivepe/newpreprocessedData/videos_128x96/1-video001_38x384x512(0).avi'
tensor=torchvision.io.read_video(str(path), output_format='TCHW')[0].float()/255
gray_tensor = torch.mean(tensor, dim=1, keepdim=True)
tensor = transform(gray_tensor)
video_tensor=torch.transpose(tensor, 0,1)'''

'''for i in range(0, 1000, 100):
    t = torch.tensor([i]).long()
    sampled, _, _ = q_sample(final_tensor, t)
    video_tensor_to_gif(sampled, f'./prova22-{i}.avi')'''
    
max_value = torch.max(video_tensor)
print('max_value', max_value)

final_tensor = cast_num_frames_fn(video_tensor)    
print(final_tensor.shape)

final_path = '/data/aolivepe/NOISE_SCHEDULE/'

noise_tensor = final_tensor

psnr_values = []
for i in range(0, 1000, 500):
    t = torch.tensor([i]).long()
    noise_tensor, _, _, betas1, betas2, betas3= q_sample(noise_tensor, t)
    #video_tensor_to_gif(sampled, f'./prova22-{i}.avi')
    #video_to_tiff(final_tensor, final_path, 'COSINE', i)
    psnr = peak_signal_noise_ratio(final_tensor.squeeze().numpy(), noise_tensor.squeeze().numpy())
    psnr_values.append(psnr)

    #video_to_png(noise_tensor, final_path, 'COSINE', i)
    
betas1 = betas1.numpy()
betas2 = betas2.numpy()
betas3 = betas3.numpy()

'''quad = np.log(make_beta_schedule('quad'))
linear = np.log(make_beta_schedule('linear'))
warmup10 = np.log(make_beta_schedule('warmup10'))
warmup50 = np.log(make_beta_schedule('warmup50'))
const = np.log(make_beta_schedule('const'))
jsd = np.log(make_beta_schedule('jsd'))'''

beta_quad = make_beta_schedule('quad')
beta_linear = make_beta_schedule('linear')
beta_jsd = make_beta_schedule('jsd')

alpha_quad = 1 -beta_quad
alpha_linear = 1 -beta_linear
alpha_jsd = 1 -beta_jsd

log_quad = np.log(alpha_quad/beta_quad)
log_linear = np.log(alpha_linear/beta_linear)
log_jsd = np.log(alpha_jsd/beta_jsd)

# Plotting cosine
'''plt.plot(range(1, len(betas1) + 1), betas1, marker='.', label=r'$\tau$ = 1')
plt.plot(range(1, len(betas2) + 1), betas2, marker='.', label=r'$\tau$ = 2')
plt.plot(range(1, len(betas3) + 1), betas3, marker='.', label=r'$\tau$ = 3')'''

# Plotting others
plt.plot(range(1, len(log_linear) + 1), log_linear, marker='.', label='linear')
plt.plot(range(1, len(log_quad) + 1), log_quad, marker='.', label='quad')
plt.plot(range(1, len(log_jsd) + 1), log_jsd, marker='.', label='jsd')



# Adding legend
plt.legend()

plt.xlabel('t')
plt.ylabel(r'log($\alpha_t$/(1-$\alpha_t$))')
plt.title('log(SNR)')
plt.grid(True)

# Save the plot as an image
plt.savefig('logotherschedule.png')