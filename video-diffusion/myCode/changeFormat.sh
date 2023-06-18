#!/bin/bash

orig_directory=./../../preprocessedData/videorecortadoScaleAVI/
fin_directory=./../../preprocessedData/videorecortadoScaleGIF/

IFS='
'
j=0
for filename in $(ls $orig_directory)
do
    ffmpeg -i $orig_directory$filename $fin_directory$j.gif
    ((j++))
done