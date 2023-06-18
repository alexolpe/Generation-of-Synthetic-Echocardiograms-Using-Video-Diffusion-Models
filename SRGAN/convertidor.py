import torch
import os
import cv2

    
os.environ['CUDA_VISIBLE_DEVICES']='3, 4'
device='cuda:0'

script_path = os.path.dirname(__file__)

# establecer la ruta del directorio que contiene los videos
videos_dir = os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'preprocessedData', 'vr480x640AVI_croped_512x384'))
output_dir = os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'preprocessedData', 'img480x640AVI_croped_512x384'))

# create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# initialize a counter for the frame number
frame_num = 0

# iterar a través de todos los archivos en el directorio
for filename in os.listdir(videos_dir):
    print(filename)
    # construir la ruta completa del archivo
    file_path = os.path.join(videos_dir, filename)

    # comprobar si el archivo es un video AVI
    if filename.endswith(".avi"):
        # abrir el archivo de video
        cap = cv2.VideoCapture(file_path)

        # loop through the frames of the video
        while cap.isOpened():
            # read a frame from the video
            ret, frame = cap.read()
            if not ret:
                break

            # construct the output file path
            output_path = os.path.join(output_dir, f'frame_{frame_num:05d}.png')
            print(f'frame_{frame_num:05d}.png')

            # write the frame as a PNG image
            cv2.imwrite(output_path, frame)

            # increment the frame counter
            frame_num += 1

        # release the video capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

