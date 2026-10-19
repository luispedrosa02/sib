from abc import ABCMeta, abstractmethod
import numpy as np
import copy


class Layer(metaclass=ABCMeta):

    @abstractmethod
    def forward_propagation(self, input, training=False):
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error):
        raise NotImplementedError

    @abstractmethod
    def output_shape(self):
        raise NotImplementedError

    @abstractmethod
    def parameters(self):
        raise NotImplementedError

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__


class DenseLayer(Layer):

    def __init__(self, n_units: int, input_shape: tuple = None):
        self.n_units = n_units
        self._input_shape = input_shape

        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

    def initialize(self, optimizer=None):
        n_features = self.input_shape()[0]

        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))

        self.weights_optimizer = copy.copy(optimizer)
        self.biases_optimizer = copy.copy(optimizer)

        return self

    def parameters(self):
        return self.weights.size + self.biases.size

    def output_shape(self):
        return (self.n_units,)

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        raise NotImplementedError(
            "A retropropagação ainda não foi implementada."
        )