#!/bin/bash

directory="/data/aolivepe/REAL_EXPERIMENTS/ATTN_A_CAPA/2_sample"  # Ruta al directorio que deseas recorrer

for file in "$directory"/*.avi; do
    if [ -f "$file" ]; then
        python3 inference_realesrgan_video.py -i "$file"
        # Hacer algo con el archivo AVI, por ejemplo, ejecutar un comando o script
    fi
done
