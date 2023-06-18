import os
import cv2
script_path = os.path.dirname(__file__)


#directory = '/../../../data/aolivepe/REAL_EXPERIMENTS/cosespetites/96x128_cropped_frame'
directory = '/../../../data/aolivepe/preprocessedData/vr480x640AVI_croped_128x96'

png_count = 0

for filename in os.listdir(directory):
    if filename.endswith(".png"):
        png_count += 1

print("Number of PNG files:", png_count)

for filename in os.listdir(directory):
    if filename.endswith(".avi"):
        file_path = os.path.join(directory, filename)
        video_capture = cv2.VideoCapture(file_path)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("File:", filename)
        print("Number of frames:", frame_count)
        video_capture.release()
