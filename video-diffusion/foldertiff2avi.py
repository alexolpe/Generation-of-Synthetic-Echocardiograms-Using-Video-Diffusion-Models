import cv2
import os

print('HOLAA')
def generar_video(subdirectorio, final_dir):
    # Obtener la lista de archivos TIFF en el subdirectorio
    archivos_tiff = [archivo for archivo in os.listdir(subdirectorio) if archivo.endswith('.tiff')]

    # Ordenar los archivos por nombre
    archivos_tiff.sort()

    # Obtener la primera imagen para obtener la información de tamaño
    imagen_inicial = cv2.imread(os.path.join(subdirectorio, archivos_tiff[0]))
    alto, ancho, _ = imagen_inicial.shape

    # Crear el objeto VideoWriter para generar el video AVI
    nombre_video = os.path.join(final_dir, f'{os.path.basename(os.path.normpath(subdirectorio))}.avi')
    print(nombre_video)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(nombre_video, fourcc, 30, (ancho, alto))

    # Leer cada archivo TIFF y agregarlo al video
    for archivo_tiff in archivos_tiff:
        ruta_imagen = os.path.join(subdirectorio, archivo_tiff)
        imagen = cv2.imread(ruta_imagen)
        video.write(imagen)

    # Liberar los recursos y cerrar el video
    video.release()

main_dir = '/data/aolivepe/dataloader_preprocessed_sample_fvd'
final_dir = '/data/aolivepe/dataloader_preprocessed_avi'

# Recorrer todos los subdirectorios y generar los videos
for directorio_actual, subdirectorios, archivos in os.walk(main_dir):
    for subdirectorio in subdirectorios:
        ruta_subdirectorio = os.path.join(directorio_actual, subdirectorio)
        generar_video(ruta_subdirectorio, final_dir)
