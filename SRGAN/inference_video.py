# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import os

import cv2
import torch
from torch import nn

import imgproc
import model_x5
from utils import load_state_dict
import numpy as np


script_path = os.path.dirname(__file__)
os.environ['CUDA_VISIBLE_DEVICES']='7'

model_names = sorted(
    name for name in model_x5.__dict__ if
    name.islower() and not name.startswith("__") and callable(model_x5.__dict__[name]))


def choice_device(device_type: str) -> torch.device:
    # Select model processing equipment type
    if device_type == "cuda":
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return device


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    sr_model = model_x5.__dict__[model_arch_name](in_channels=3,
                                               out_channels=3,
                                               channels=64,
                                               num_rcb=16)
    sr_model = sr_model.to(device=device)

    return sr_model

def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    array = tensor.detach().cpu().numpy()

    # reshape the numpy array to video dimensions
    array = np.transpose(array, (0, 2, 3, 1))
    print(array.shape)
    array = np.uint8(array * 255)

    # create video using OpenCV
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (tensor.shape[3], tensor.shape[2]))
    for i in range(tensor.shape[0]):
        out.write(array[i])
        print(i)
    out.release()
    return 

def main(args):
    device = choice_device(args.device_type)
    
    # Initialize the model
    sr_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    sr_model = load_state_dict(sr_model, args.model_weights_path)

    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Start the verification mode of the model.
    sr_model.eval()
    
    videos_dir = args.inputs_path
    # iterar a través de todos los archivos en el directorio
    for filename in os.listdir(videos_dir):
        video_tensor_list=[]
        
        # construir la ruta completa del archivo
        file_path = os.path.join(videos_dir, filename)
        print('file_path',file_path)

        # comprobar si el archivo es un video AVI
        if filename.endswith(".avi"):

            # abrir el archivo de video
            cap = cv2.VideoCapture(file_path)

            num_frame=0

            # loop through the frames of the video
            while cap.isOpened():
                # read a frame from the video
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = frame.astype(np.float32) / 255.
                #for i in range(8):
                #frame = cv2.imread(f'{args.inputs_path}/frame_{i:05d}.png').astype(np.float32) / 255.

                print('frame ', frame.dtype)

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Convert image data to pytorch format data
                tensor = imgproc.image_to_tensor(image, False, False).unsqueeze_(0)

                # Transfer tensor channel image format data to CUDA device
                lr_tensor = tensor.to(device, non_blocking=True)

                # Use the model to generate super-resolved images
                with torch.no_grad():
                    sr_tensor = sr_model(lr_tensor)

                print('sr_tensor', sr_tensor.shape)
                video_tensor_list.append(sr_tensor)

                num_frame+=1

            # release the video capture and close all windows
            cap.release()
            cv2.destroyAllWindows()

            video_tensor = torch.cat(video_tensor_list, dim=0)
            print('video_tensor', video_tensor.shape)
            output_filename = '.'.join(filename.rsplit('.', 1)[:-1]) + '_SR.avi'
            out_path = os.path.join(args.output_path, output_filename)
            video_tensor_to_gif(video_tensor, out_path)
            print(f"SR video save to `{out_path}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Using the model generator super-resolution images.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="srresnet_x4")
    parser.add_argument("--inputs_path",
                        type=str,
                        default=os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'newpreprocessedData', 'psnr_SRGAN_2')),
                        #default=os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'preprocessedData', 'vr96x128AVI', 'video0.avi')),
                        help="Low-resolution image path.")
    parser.add_argument("--output_path",
                        type=str,
                        default=os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'newpreprocessedData', 'psnr_SRGAN_2')),
                        help="Super-resolution image path.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./../../../data/aolivepe/srgan_results/results/newECG_pretrained_gen_last/g_best.pth.tar",
                        #default="./results/newECG_pretrained_gen_last/g_best.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--device_type",
                        type=str,
                        default="cpu",
                        choices=["cpu", "cuda"])
    args = parser.parse_args()

    main(args)
