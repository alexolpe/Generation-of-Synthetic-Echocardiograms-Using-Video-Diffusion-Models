# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code that computes FVD for some empty frames.

The FVD for this setup should be around 131.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import frechet_video_distance as fvd
import os
import random
import glob
import cv2

# Number of videos must be divisible by 16.
NUMBER_OF_VIDEOS = 100
VIDEO_LENGTH = 15

os.environ['CUDA_VISIBLE_DEVICES']='7'


def get_video_tensor(folder_path):
  # Get a list of all video file names in the folder
  video_files = [file for file in os.listdir(folder_path) if file.endswith('.gif')]

  # Randomly select 16 video files from the list
  random_files = random.sample(video_files, NUMBER_OF_VIDEOS)

  # Read and convert the videos into TensorFlow tensors
  video_tensors = []
  for file in random_files:
      video_path = os.path.join(folder_path, file)
      video_tensor = tf.image.decode_gif(tf.io.read_file(video_path))
      video_tensors.append(video_tensor)

  # Concatenate the video tensors into a single TensorFlow tensor
  video_tensor = tf.stack(video_tensors)
  return video_tensor

def get_tensor(path):
  video_file_paths = glob.glob(path)
  print('ordered', len(video_file_paths), type(video_file_paths))
  #video_file_paths = os.listdir(video_file_paths)
  random.shuffle(video_file_paths)
  print('shuffled', len(video_file_paths), type(video_file_paths))

  videos = []
  num_videos = 1
  random_number = random.randint(1, 2)
  for video_path in video_file_paths:
    jj = random_number+NUMBER_OF_VIDEOS
    if num_videos >= 50 and num_videos < 200:
    #if num_videos <= 150:
      video = cv2.VideoCapture(video_path)
      video_frames = []
      
      while True:
          ret, frame = video.read()
          if not ret:
              break
          f=tf.convert_to_tensor(frame, dtype=tf.float32)
          video_frames.append(f)
      
      video_tensor = tf.stack(video_frames)
      videos.append(video_tensor)
    num_videos += 1
  tensor = tf.stack(videos)
  return tensor

def main(argv):
  # Set the path to the folder containing the videos
  real_path = '/data/aolivepe/newpreprocessedData/videos_512x384/*.avi'
  fake_path = '/data/aolivepe/REAL_EXPERIMENTS/SUPER_RESOLUTION/SRGAN/*.avi'
  
  del argv
  with tf.Graph().as_default():

    '''first_set_of_videos = get_video_tensor(real_path)
    print('first_set_of_videos', first_set_of_videos.shape)
    second_set_of_videos = get_video_tensor(fake_path)
    print('second_set_of_videos', second_set_of_videos.shape)'''
    
    first_set_of_videos_path = '/../../../data/aolivepe/newpreprocessedData/originalgif_64x64/*.gif'
    second_set_of_videos_path = '/../../../data/aolivepe/newpreprocessedData/originalgif_64x64/*.gif'
    
    #first_set_of_videos = tf.zeros([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3])
    #second_set_of_videos = tf.ones([NUMBER_OF_VIDEOS, VIDEO_LENGTH, 64, 64, 3]) * 255
    first_set_of_videos=get_tensor(real_path)
    print('first_set_of_videos', first_set_of_videos.shape)
    second_set_of_videos=get_tensor(fake_path)
    print('second_set_of_videos', second_set_of_videos.shape)

    result = fvd.calculate_fvd(
        fvd.create_id3_embedding(fvd.preprocess(first_set_of_videos,
                                                (224, 224))),
        fvd.create_id3_embedding(fvd.preprocess(second_set_of_videos,
                                                (224, 224))))

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      print("FVD is: %.2f." % sess.run(result))


if __name__ == "__main__":
  tf.app.run(main)
