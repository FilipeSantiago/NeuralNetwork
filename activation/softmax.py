import numpy as np


class Softmax:

    def __init__(self):
        self.output = None
        self.max_values = []
        self.sums_exp = []

    def forward(self, inputs):
        self.max_values = np.max(inputs, axis=1, keepdims=True)

        # guarantees that the maximum value passed to the exponential function is 0, preventing overflow
        exp_inputs = inputs - self.max_values

        # runs an exponential function to deal with negative numbers without losing information
        exp_values = np.exp(exp_inputs)

        # sums each vector keeping a compatible shape with exp_values [5, 3] => [5, 1]
        self.sums_exp = np.sum(exp_values, axis=1, keepdims=True)

        # normalize vectors
        probabilities = exp_values / self.sums_exp

        self.output = probabilities
        return self.output

    def backward(self, grad):
        # well explained in https://www.youtube.com/watch?v=M59JElEPgIg
        return self.output * (grad - (grad * self.output).sum(axis=1)[:, None])
