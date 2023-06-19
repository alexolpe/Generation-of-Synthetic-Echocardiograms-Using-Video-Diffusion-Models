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
import argparse

# Number of videos must be divisible by 16.
NUMBER_OF_VIDEOS = 100
VIDEO_LENGTH = 15

os.environ['CUDA_VISIBLE_DEVICES']='7'

def get_tensor(path):
  video_file_paths = glob.glob(path)
  print('ordered', len(video_file_paths), type(video_file_paths))
  random.shuffle(video_file_paths)
  print('shuffled', len(video_file_paths), type(video_file_paths))

  videos = []
  num_videos = 1
  random_number = random.randint(1, 2)
  for video_path in video_file_paths:
    jj = random_number+NUMBER_OF_VIDEOS
    if num_videos >= 50 and num_videos < 200:
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
  real_path = os.path.join(FLAGS.real, '*.avi')
  fake_path = os.path.join(FLAGS.fake, '*.avi')
  '''real_path = '/data/aolivepe/newpreprocessedData/videos_512x384/*.avi'
  fake_path = '/data/aolivepe/REAL_EXPERIMENTS/SUPER_RESOLUTION/SRGAN/*.avi'''
  
  del argv
  with tf.Graph().as_default():

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
  parser = argparse.ArgumentParser(description='Example argument parser')

  # Add arguments
  parser.add_argument('--real', help='Real file path')
  parser.add_argument('--fake', help='Fake file path')

  # Parse the arguments
  FLAGS, _ = parser.parse_known_args()

  # Call the main function using tf.app.run()
  tf.app.run(main=main)
