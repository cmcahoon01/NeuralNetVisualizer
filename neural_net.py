import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return np.where(x > 0,
                    1 / (1 + np.power(10, -x)),
                    np.power(10, x) / (1 + np.power(10, x)))


def relu_approximation(x, derivative=False):
    if derivative:
        return sigmoid(x)
    print(x)

    return np.where(x > 0,
                    np.log10(1 + np.power(10, -x)) + np.maximum(x, 0),
                    np.log10(1 + np.power(10, x)))


class NeuralNet:
    class Layer:
        input = None
        output = None
        middle = None
        error = None
        delta = None

        def __init__(self, n_inputs, n_neurons, activation_function):
            self.weights = np.random.randn(n_inputs, n_neurons)
            self.biases = np.random.randn(1, n_neurons)
            self.activation_function = activation_function

        def forward(self, inputs):
            self.input = inputs
            self.middle = (np.dot(self.input, self.weights) + self.biases).flatten()
            self.output = self.activation_function(self.middle)
            return self.output

    def __init__(self, inputs, outputs, hidden_layers, hidden_nodes, activation_function):
        self.num_inputs = inputs
        self.num_outputs = outputs
        self.activation_function = activation_function

        self.layers = []
        previous_layer_size = inputs
        for i in range(hidden_layers):
            self.layers.append(NeuralNet.Layer(previous_layer_size, hidden_nodes, activation_function))
            previous_layer_size = hidden_nodes
        self.layers.append(NeuralNet.Layer(previous_layer_size, outputs, activation_function))

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, inputs, outputs, epochs, learning_rate, alpha=1, visualize=None):
        rolling_error = 0.5
        for epoch in range(epochs):
            avg_error = 0
            for i in range(len(inputs)):
                self.forward(inputs[i])
                self.backprop(outputs[i], learning_rate)
                current_error = abs(np.sum(self.layers[-1].error))
                avg_error += current_error
                if visualize is not None and (epoch + 990) % 1000 == 0:
                    visualize.visualize(self)
            rolling_error = 0.9 * rolling_error + 0.1 * avg_error / len(inputs)
            if epoch % 500 == 0:
                print("Epoch:", epoch, "Error:", round(rolling_error, 5), "Learning Rate:", round(learning_rate, 5))
            learning_rate = learning_rate * (1 - alpha)

    def backprop(self, expected_output, learning_rate):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                layer.error = (layer.output - expected_output)  # derivative of loss function, (y - t)^2 / 2
                layer.delta = layer.error * self.activation_function(layer.middle, derivative=True)
            else:
                next_layer = self.layers[i + 1]
                layer.error = np.dot(next_layer.delta, next_layer.weights.T)
                layer.delta = layer.error * self.activation_function(layer.middle, derivative=True)

            front_values = np.expand_dims(layer.input, axis=0)
            back_values = np.expand_dims(layer.delta, axis=1)

            change_in_weight = np.dot(back_values, front_values).T

            layer.weights -= learning_rate * change_in_weight
            layer.biases -= learning_rate * layer.delta
