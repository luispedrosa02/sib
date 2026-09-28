import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    The RidgeRegression is a linear model using L2 regularization.
    This model solves the linear regression problem using Gradient Descent.

    Parameters
    ----------
    l2_penalty: float, default=1.0
        The L2 regularization parameter.
    alpha: float, default=0.001
        The learning rate.
    max_iter: int, default=1000
        The maximum number of iterations.
    patience: int, default=5
        The maximum number of iterations without improvement allowed.
    scale: bool, default=True
        Whether to scale the dataset or not.
    """

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

        # Estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset: Dataset) -> 'RidgeRegression':
        if self.scale:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            self.std[self.std == 0] = 1.0
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = X.shape
        self.theta = np.zeros(n)
        self.theta_zero = 0.0

        i = 0
        early_stopping = 0

        while i < self.max_iter and early_stopping < self.patience:
            # Predict
            y_pred = np.dot(X, self.theta) + self.theta_zero

            # Error
            error = y_pred - dataset.y

            # Gradients
            grad_theta_zero = (1 / m) * np.sum(error)
            grad_theta = (1 / m) * np.dot(X.T, error) + (self.l2_penalty / m) * self.theta

            # Parameter updates
            self.theta_zero -= self.alpha * grad_theta_zero
            self.theta -= self.alpha * grad_theta

            # Cost
            current_cost = self.cost(Dataset(X=X, y=dataset.y))
            self.cost_history[i] = current_cost

            # Early stopping check
            if i > 0 and self.cost_history[i - 1] - current_cost < 1e-4:
                early_stopping += 1
            else:
                early_stopping = 0

            i += 1

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return mse(dataset.y, predictions)

    def cost(self, dataset: Dataset) -> float:
        y_pred = self.predict(dataset)
        m = dataset.shape()[0]
        cost_mse = np.sum((y_pred - dataset.y) ** 2)
        cost_reg = self.l2_penalty * np.sum(self.theta ** 2)
        return float((1 / (2 * m)) * (cost_mse + cost_reg))