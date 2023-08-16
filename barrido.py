import numpy as np #operaciones numericas
import matplotlib.pyplot as plt #visualizacion en pantalla
import matplotlib.colors as mcolors #manipulacion de colores para el mapa de colores

class Barrido:
    def __init__(self, input_size, hidden_size, output_size):
        # Define la arquitectura de la red neuronal
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Genera los pesos aleatorios para las conexiones entre neuronas
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Define la función de activación para las neuronas
        self.activation_function = self.sigmoid

    def sigmoid(self, x):
        # Función de activación sigmoidal
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        # Propagación hacia adelante (Forward propagation)
        hidden_layer_input = np.dot(inputs, self.weights_input_hidden)
        hidden_layer_output = self.activation_function(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output_layer_output = self.activation_function(output_layer_input)

        return output_layer_output

def generate_random_points(num_points, output_file):
    # Genera puntos aleatorios en el rectángulo [0, 10] x [0, 10]
    points = np.random.rand(num_points, 2) * 10
    return points

def custom_colormap():
    # Definir los colores y las posiciones de transición en el colormap
    colors = [(1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]  # Blanco a Rojo
    positions = [0.0, 1.0]

    # Crear el colormap personalizado
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', colors, N=256)

    return cmap

def plot_classification(image):
    # Obtener el colormap personalizado (rojo y blanco)
    cmap = custom_colormap()

    # Mostrar la imagen utilizando el colormap personalizado
    plt.imshow(image, extent=[0, 10, 0, 10], cmap=cmap)

    # Resto del código para configurar el gráfico (etiquetas, título, colorbar)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Imagen generada por la red neuronal')
    plt.colorbar(ticks=[-1, 0, 1], label='Clasificación')
    plt.show()
    
def main():
    input_size = 2  # Dos dimensiones (coordenadas x, y)
    hidden_size = 5  # Número de neuronas en la capa oculta
    output_size = 1  # Una dimensión para la clasificación binaria (-1 o 1)

    neural_network = Barrido(input_size, hidden_size, output_size)

    # Genera puntos aleatorios en el rectángulo [0, 10] x [0, 10]
    points = generate_random_points(1000, "points.csv")

    # Clasifica los puntos con la red neuronal y crea una imagen
    image = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            x = i / 100 * 10
            y = j / 100 * 10
            point = np.array([x, y])
            classification = neural_network.predict(point)
            image[i, j] = classification[0]

    # Grafica la imagen generada por la red neuronal
    plot_classification(image)

if __name__ == "__main__":
    main()
