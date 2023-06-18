import cv2
import os

orig_directory="../../preprocessedData/videosolo/"
fin_directory="../../preprocessedData/videorecortadoScale/"

directory= fin_directory

for file in os.listdir(directory):
    cap=cv2.VideoCapture(directory+file)
    print("Number of frames "+file+": "+str(cap.get(7)))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Dimension "+file+": "+str(frame_width)+" "+str(frame_height))