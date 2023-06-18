import os
import imageio
script_path = os.path.dirname(__file__)

# Directorio que contiene los videos
#directory = os.path.abspath(os.path.join(script_path, '..', '..', '..','..', 'data', 'aolivepe', 'REAL_EXPERIMENTS', 'SUPER_RESOLUTION', 'SRGAN'))
directory = "/data/aolivepe/newpreprocessedData/psnr_Real-ESRGAN_2"
# Directorio de salida para guardar las imágenes PNG
#output_directory = os.path.abspath(os.path.join(script_path, '..', '..', '..','..', 'data', 'aolivepe', 'REAL_EXPERIMENTS', 'SUPER_RESOLUTION', 'SRGAN_frames'))
output_directory = "/data/aolivepe/newpreprocessedData/psnr_Real-ESRGAN_2"
# Verificar si el directorio de salida existe, si no, crearlo
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Recorrer los archivos en el directorio
for filename in os.listdir(directory):
    if filename.endswith(".avi"):
        # Ruta completa del archivo GIF
        gif_path = os.path.join(directory, filename)

        # Leer el archivo GIF
        gif_frames = imageio.mimread(gif_path)

        # Iterar a través de los frames del GIF
        for i, frame in enumerate(gif_frames):
            # Construir el nombre del archivo de salida
            output_filename = f"{os.path.splitext(filename)[0]}_{i}.png"

            # Ruta completa del archivo de salida
            output_path = os.path.join(output_directory, output_filename)

            # Guardar el frame como imagen PNG
            imageio.imwrite(output_path, frame)