import matplotlib.pyplot as plt
import os

script_path = os.path.dirname(__file__)

def leer_archivo(nombre_archivo):
    valores_x = []
    valores_y = []

    with open(nombre_archivo, 'r') as archivo:
        for linea in archivo:
            datos = linea.strip().split(':')
            if len(datos) == 2:
                valor_x = float(datos[0].strip())
                valor_y = float(datos[1].strip())
                valores_x.append(valor_x)
                valores_y.append(valor_y)

    return valores_x, valores_y

def graficar(valores_x, valores_y):
    plt.plot(valores_x, valores_y, 'o-')
    plt.xlabel('Valor X')
    plt.ylabel('Valor Y')
    plt.title('Representación gráfica')
    plt.grid(True)
    plt.savefig(os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'REAL_EXPERIMENTS', 'original_b4_96x128_avi_Att_in_the_last_layer', 'grafica.png')))  # Guardar la gráfica en un archivo PNG
    plt.show()

# Nombre del archivo que contiene los datos
nombre_archivo = os.path.abspath(os.path.join(script_path, '..', '..', '..', 'data', 'aolivepe', 'REAL_EXPERIMENTS', 'original_b4_96x128_avi_Att_in_the_last_layer', 'loss'))

# Leer los valores del archivo
valores_x, valores_y = leer_archivo(nombre_archivo)

print(len(valores_x))
# Graficar los valores
graficar(valores_x, valores_y)
