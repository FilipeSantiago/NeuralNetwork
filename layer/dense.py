import numpy as np


class Dense:

    def __init__(self, n_inputs, n_neurons, activation):
        self.n_inputs = n_inputs
        self.activation = activation
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros(n_neurons)
        self.output = None
        self.inputs = None
        self.n = n_neurons

        self.db = 0
        self.dw = 0
        self.dz = 0
        self.da = 0

    def forward(self, inputs):
        """
        Calculates the values of the current layer

        :param inputs: values of the previous layer neurons
        """

        self.inputs = np.array(inputs)
        weights_arr = np.array(self.weights)
        layer_base_forward = np.dot(inputs, weights_arr) + self.biases
        self.output = self.activation.forward(layer_base_forward)

    def backward(self, input_da):

        weights_arr = np.array(self.weights)

        dz = self.activation.backward(input_da)

        dw = 1. / self.n * self.inputs.T.dot(dz)
        db = 1. / self.n * np.sum(dz, axis=0, keepdims=True)

        self.da = weights_arr.dot(dz.T).T
        self.db = db
        self.dw = dw
        self.dz = dz

    def update_parameters(self, learning_rate):
        self.weights = self.weights - learning_rate * self.dw
        self.biases = self.biases - learning_rate * self.db
