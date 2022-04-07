import numpy as np

from neural_net import NeuralNet, relu_approximation, sigmoid
from visualize import start_pygame

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # XOR
y = np.array([[0], [1], [1], [0]])

VISUALIZE = True
SHOW_VALUES = False


def main():
    window_size = (1800, 600)

    visualize = start_pygame(window_size, SHOW_VALUES) if VISUALIZE else None

    net = NeuralNet(inputs=2, outputs=1, hidden_layers=1, hidden_nodes=2, activation_function=sigmoid)

    net.train(X, y, epochs=100000, learning_rate=1, alpha=0, visualize=visualize)


if __name__ == '__main__':
    main()
