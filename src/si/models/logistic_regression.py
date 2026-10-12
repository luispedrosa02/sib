import numpy as np

from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """Regressão logística binária com regularização L2."""

    def __init__(
        self,
        l2_penalty: float = 1.0,
        alpha: float = 0.001,
        max_iter: int = 1000,
        patience: int = 5,
        scale: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> "LogisticRegression":
        if self.scale:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            self.std[self.std == 0] = 1.0
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        y = dataset.y
        m, n = X.shape

        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}
        early_stopping = 0

        for i in range(self.max_iter):
            probabilities = sigmoid_function(
                X @ self.theta + self.theta_zero
            )
            error = probabilities - y

            grad_theta = (
                X.T @ error / m
                + self.l2_penalty * self.theta / m
            )
            grad_theta_zero = np.mean(error)

            self.theta -= self.alpha * grad_theta
            self.theta_zero -= self.alpha * grad_theta_zero

            current_cost = self.cost(dataset)
            self.cost_history[i] = current_cost

            if i > 0:
                improvement = self.cost_history[i - 1] - current_cost
                if improvement < 1e-4:
                    early_stopping += 1
                else:
                    early_stopping = 0

            if early_stopping >= self.patience:
                break

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        probabilities = sigmoid_function(
            X @ self.theta + self.theta_zero
        )
        return (probabilities >= 0.5).astype(int)

    def _score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)

    def cost(self, dataset: Dataset) -> float:
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        probabilities = sigmoid_function(
            X @ self.theta + self.theta_zero
        )

        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        y = dataset.y
        m = X.shape[0]

        cross_entropy = -np.mean(
            y * np.log(probabilities)
            + (1 - y) * np.log(1 - probabilities)
        )

        regularization = (
            self.l2_penalty / (2 * m)
        ) * np.sum(self.theta ** 2)

        return float(cross_entropy + regularization)