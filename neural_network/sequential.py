from loss import CrossEntropy
import numpy as np
from sklearn.metrics import accuracy_score


class Sequential:

    def __init__(self, *layers):
        self.layers = layers

    def start_learning(self, X, y, epochs=100, batch_size=32, learning_rate=0.1, loss=CrossEntropy()):

        for i in range(0, epochs):
            for batch, batch_y in self.__group_list(X, y, int(batch_size)):
                self.__forward(batch)

                loss_f = loss.forward(self.layers[-1].output, batch_y)
                loss_b = loss.backward(self.layers[-1].output, batch_y)

                self.__backward(loss_b)
                self.__learn(learning_rate)

                y_pred = np.argmax(self.layers[-1].output, axis=1)
                y_true = np.argmax(batch_y.values, axis=1)

                print(accuracy_score(y_true, y_pred))

    def predict(self, X):
        return self.__forward(X)

    def __forward(self, X):
        data = X
        for layer in self.layers:
            layer.forward(data)
            data = layer.output

        return data

    def __backward(self, loss_b):
        da = None
        for j in range(0, len(self.layers)):
            i = len(self.layers) - j - 1
            if i == len(self.layers) - 1:
                self.layers[i].backward(loss_b)
            else:
                self.layers[i].backward(da)
            da = self.layers[i].da

    def __learn(self, learning_rate):
        for layer in self.layers:
            layer.update_parameters(learning_rate)

    @staticmethod
    def __group_list(l, l2, group_size):
        """
        :param l:           list
        :param group_size:  size of each group
        :return:            Yields successive group-sized lists from l.
        """
        for i in range(0, len(l), group_size):
            yield l[i:i + group_size], l2[i:i + group_size]
