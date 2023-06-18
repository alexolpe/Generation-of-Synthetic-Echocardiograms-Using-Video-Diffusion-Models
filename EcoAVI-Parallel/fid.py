from torchmetrics.image.fid import FrechetInceptionDistance

real_images =
fake_images = 

import os
import cv2
import torch

# Set directory path
dir_path = "/path/to/videos/directory/"

# Get list of video files in directory
video_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.mp4')]

# Initialize empty list to store frames
frames = []

# Loop through each video file
for video_file in video_files:
    # Open video file with OpenCV
    cap = cv2.VideoCapture(video_file)

    # Loop through each frame in video
    while True:
        # Read frame from video
        ret, frame = cap.read()

        # If end of video, break loop
        if not ret:
            break

        # Convert frame to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Add frame to list of frames
        frames.append(frame_gray)

    # Release OpenCV video capture object
    cap.release()

# Convert list of frames to tensor
real_images = torch.tensor(frames)

# Add channel dimension to frames tensor
real_images = real_images.unsqueeze(1)

print(real_images.shape)


fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")
# FID: 177.7147216796875