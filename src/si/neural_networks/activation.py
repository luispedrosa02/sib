import numpy as np
from abc import abstractmethod

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):

    def __init__(self, input_shape=None):
        self._input_shape = input_shape
        self.input = None
        self.output = None

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return output_error * self.derivative(self.input)

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0

class ReLUActivation(ActivationLayer):

    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return (input > 0).astype(float)


class SigmoidActivation(ActivationLayer):

    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        sigmoid = self.activation_function(input)
        return sigmoid * (1 - sigmoid)

class TanhActivation(ActivationLayer):

    def activation_function(self, input):
        return np.tanh(input)

    def derivative(self, input):
        return 1 - self.activation_function(input) ** 2


class SoftmaxActivation(ActivationLayer):

    def activation_function(self, input):
        shifted = input - np.max(
            input, axis=-1, keepdims=True
        )

        exp_shifted = np.exp(shifted)

        return exp_shifted / np.sum(
            exp_shifted, axis=-1, keepdims=True
        )

    def derivative(self, input):
        # Matriz com as derivadas de todas as saídas
        output = self.activation_function(input)
        identity = np.eye(output.shape[-1])

        return output[..., :, None] * (
            identity - output[..., None, :]
        )

    def backward_propagation(self, output_error):
        weighted_sum = np.sum(
            output_error * self.output,
            axis=-1,
            keepdims=True
        )

        return self.output * (output_error - weighted_sum)