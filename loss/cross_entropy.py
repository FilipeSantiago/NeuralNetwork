import numpy as np

from loss.loss import Loss


class CrossEntropy(Loss):

    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        y_true = np.array(y_true)
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = []

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, y_pred, y_true):
        y_true = np.array(y_true)
        return y_pred - y_true
