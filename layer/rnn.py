import numpy as np


class RNN:

    def __init__(self, rnn_units, input_dim, output_dim, activation):
        # np.random.seed()
        self.rnn_units = rnn_units
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hx = 0.1 * np.random.randn(rnn_units, input_dim)
        self.hh = 0.1 * np.random.randn(rnn_units, rnn_units)
        self.yh = 0.1 * np.random.randn(output_dim, rnn_units)

        self.activation = activation

    def forward_recurrent(self, input_unity, h):
        h = self.activation(self.hh * h + self.hx * input_unity)
        output = self.yh * h

        return output, h

    def forward(self, inputs):
        h = np.zeros(self.rnn_units)
        output = np.zeros(self.output_dim)

        for input_unity in inputs:
            output, h = self.forward_recurrent(input_unity, h)

        return output

    def backward(self, input_da):
        pass

    def update_parameters(self, learning_rate):
        pass
