import argparse
import cv2
import os
import math

orig_directory="../../preprocessedData/videosolo/"
fin_directory="../../preprocessedData/videorecortadoScaleAVI/"

box_dim = 128 
y_roi=4
x_roi=4

starting_frame = 0
ending_frame = []
num_frames_fin = 8

k=0
for file in os.listdir(orig_directory):
    cap = cv2.VideoCapture(orig_directory+file) 
    
    roi_size = (box_dim, box_dim)    
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames_orig = cap.get(7)
    num_videos = int(num_frames_orig/num_frames_fin)
    print(num_videos)
    
    for j in range(num_videos):
        fourcc=cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(fin_directory+file+str(k)+".avi", fourcc, fps, roi_size) 
        frames_progression = 0
        
        starting_frame=num_videos*num_frames_fin
        
        for i in range(num_frames_fin): 
            cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame-1)
            ret, frame = cap.read() 
            if ret == True:
                resized=cv2.resize(frame,(136,136))
                roi = resized[y_roi:y_roi+box_dim, x_roi:x_roi+box_dim]
                #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                (row, col) = roi.shape[0:2]
                for a in range(row):
                    for b in range(col):
                        # Find the average of the BGR pixel values
                        roi[a, b] = sum(roi[a, b]) * 0.33
                out.write(roi)    
            else:
                print("Cannot retrieve frames. Breaking.") #If a frame cannot be retrieved, this error is thrown
            if (out.isOpened() == False):
                print("Error opening the video file") # If the out video cannot be opened, this error is thrown
            else:
                frames_progression = frames_progression + 1 # Shows how far the frame writting process got. Compare this to the desired frame length
        print(frames_progression)
        out.release()
        k+=1

    cap.release()
    cv2.destroyAllWindows()
    print("Finished writing new video "+file)
