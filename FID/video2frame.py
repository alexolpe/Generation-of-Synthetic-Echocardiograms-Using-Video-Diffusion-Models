import os
import imageio
import argparse

script_path = os.path.dirname(__file__)

#directory = "/data/aolivepe/newpreprocessedData/psnr_Real-ESRGAN_2"
#output_directory = "/data/aolivepe/newpreprocessedData/psnr_Real-ESRGAN_2"

parser = argparse.ArgumentParser(description='Example argument parser')

parser.add_argument('--input', help='Input AVI file path')
parser.add_argument('--output', help='Output PNG file path')

args = parser.parse_args()

directory = args.input
output_directory = args.output


if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for filename in os.listdir(directory):
    if filename.endswith(".avi"):
        gif_path = os.path.join(directory, filename)

        gif_frames = imageio.mimread(gif_path)

        for i, frame in enumerate(gif_frames):
            output_filename = f"{os.path.splitext(filename)[0]}_{i}.png"

            output_path = os.path.join(output_directory, output_filename)

            imageio.imwrite(output_path, frame)
            
