import torch
from video_diffusion import Unet3D, GaussianDiffusion
import os
import cv2
import numpy as np
import torch
from PIL import Image
from einops import rearrange

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

model_path = '/data/aolivepe/test/model-1.pt'
unet.load_state_dict(torch.load(model_path))
output = "/data/aolivepe/cos_tau_3_sample"

def tensor_to_video(tensor, path):
    array = tensor.detach().cpu().numpy()

    array = np.transpose(array, (1, 2, 3, 0))
    array = np.uint8(array * 255)

    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (tensor.shape[3], tensor.shape[2]), False)
    for i in range(tensor.shape[1]):
        out.write(array[i])
    out.release()
    return
        
def tensor_to_frames(video_tensor, output_folder, num_video):
    num_canales, num_frames, altura, anchura = video_tensor.size()
    for i in range(num_frames):
        frame = video_tensor[:, i, :, :]
        print('frame', frame.shape)
        
        normalized_frame = (frame - frame.min()) / (frame.max() - frame.min())
        normalized_frame = (normalized_frame * 255).byte()
    
        normalized_frame = normalized_frame.squeeze()
        print('frame', normalized_frame.shape)
        imagen = Image.fromarray(normalized_frame.cpu().byte().numpy(), mode='L')
        image_path = f"{output_folder}/{num_video}-frame_{i}.png"
        imagen.save(image_path)


for n in reversed(range(224)):
    num_sample_rows = 1

    num_samples = num_sample_rows ** 2 #A: num_samples=16
    batches = [1]

    all_videos_list = list(map(lambda n: diffusion.sample(unet = unet, batch_size=n), batches))
    all_videos_list = torch.cat(all_videos_list, dim = 0)

    one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)', i = num_sample_rows)
    
    #SAMPLE FRAMES
    tensor_to_frames(one_gif, output, str(n))
    
    #SAMPLE VIDEOS
    video_path = os.path.join(output, f'{n}.avi')
    tensor_to_video(one_gif, video_path)