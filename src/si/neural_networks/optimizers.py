from abc import ABCMeta, abstractmethod

import numpy as np


class Optimizer(metaclass=ABCMeta):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w, grad_loss_w):
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)

        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w, grad_loss_w):
        # Inicializar o histórico na primeira atualização
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros_like(w)

        # Combinar o histórico com o gradiente atual
        self.retained_gradient = (
            self.momentum * self.retained_gradient
            + (1 - self.momentum) * grad_loss_w
        )

        # Devolver os parâmetros atualizados
        return w - self.learning_rate * self.retained_gradient

class Adam(Optimizer):

    def __init__(self, learning_rate=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, w, grad_loss_w):
        if self.m is None or self.v is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1

        # Slides 17 e 18: atualizar as medias moveis
        self.m = (
            self.beta_1 * self.m
            + (1 - self.beta_1) * grad_loss_w
        )

        self.v = (
            self.beta_2 * self.v
            + (1 - self.beta_2) * grad_loss_w ** 2
        )

        # Slide 19: compensar a inicializacao em zero
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Slide 19: atualizar os parametros
        return w - self.learning_rate * m_hat / (
            np.sqrt(v_hat) + self.epsilon
        )