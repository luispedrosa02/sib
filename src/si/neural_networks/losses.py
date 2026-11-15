from abc import ABCMeta, abstractmethod

import numpy as np


class LossFunction(metaclass=ABCMeta):

    @abstractmethod
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        raise NotImplementedError


class MeanSquaredError(LossFunction):

    def loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

class BinaryCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.sum(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return (
            -y_true / y_pred
            + (1 - y_true) / (1 - y_pred)
        )

class CategoricalCrossEntropy(LossFunction):

    def loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -np.sum(y_true * np.log(y_pred))

    def derivative(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return -y_true / y_pred