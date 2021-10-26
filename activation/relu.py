import numpy as np


class Relu:

    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, grad):
        # Se > 0 = 1, otherwise 0
        return np.where(self.output > 0, grad, 0)
