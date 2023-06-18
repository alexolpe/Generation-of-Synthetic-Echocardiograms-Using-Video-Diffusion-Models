#!/bin/bash

orig_directory=../../preprocessedData/videosolo/
fin_directory=../../preprocessedData/videorecortadoNoScale/

num_frames_fin=8
height_fin=128
width_fin=128

IFS='
'
j=0
for filename in $(ls $orig_directory)
do
    #marranada per obtenir metadades del fitxer original

    #the following line does not work here. Take a look at it. In my laptop it works
    num_frames_orig=$(ffmpeg -i $orig_directory$filename -vcodec copy -f rawvideo -y /dev/null 2>&1 | tr ^M '\n' | awk '/^frame=/ {print $2}'|tail -n 1 | tee /dev/tty)
    height_orig=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=s=x:p=0 $orig_directory$filename| tee /dev/tty)
    width_orig=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=s=x:p=0 $orig_directory$filename| tee /dev/tty)

    let num_videos=($num_frames_orig/$num_frames_fin)
    
    i=0
    while [ $i -lt $num_videos ]
    do
        echo $filename
        let init_frame=$i*$num_frames_fin
        let x_crop=$(($width_orig/2))-$(($width_fin/2))
        let y_crop=$(($height_orig/2))-$(($height_fin/2))

        #treure
        rm $fin_directory"video"$j.gif

        #millorar el crop
        ffmpeg -i $orig_directory$filename -filter:v "crop=$width_fin:$height_fin:$x_crop:$y_crop, select=gte(n\,$init_frame)" -vframes $num_frames_fin $fin_directory"video"$j.gif
        ((j++))
        ((i++))
    done
done