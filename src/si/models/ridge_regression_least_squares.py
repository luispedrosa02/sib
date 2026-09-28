import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Linear Regression model with L2 regularization using the analytical Least Squares method.

    Parameters
    ----------
    l2_penalty: float, default=1.0
        The L2 regularization parameter (lambda).
    scale: bool, default=True
        Whether to scale the dataset (standardization) or not.
    """

    def __init__(self, l2_penalty: float = 1.0, scale: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        # Estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':
        # 1. Scale data if required
        if self.scale:
            self.mean = np.mean(dataset.X, axis=0)
            self.std = np.std(dataset.X, axis=0)
            self.std[self.std == 0] = 1.0
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = X.shape

        # 2. Add intercept term (column of ones in the first column)
        X_with_intercept = np.c_[np.ones(m), X]

        # 3. Compute penalty matrix (l2_penalty * identity matrix)
        penalty_matrix = self.l2_penalty * np.eye(n + 1)

        # 4. Set first position to 0 so theta_zero is not penalized
        penalty_matrix[0, 0] = 0.0

        # 5. Compute theta parameters using the normal equation: inv(X^T @ X + penalty) @ X^T @ y
        thetas = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept) + penalty_matrix).dot(X_with_intercept.T).dot(dataset.y)

        # Split theta_zero and feature thetas
        self.theta_zero = float(thetas[0])
        self.theta = thetas[1:]

        return self

    def _predict(self, dataset: Dataset) -> np.ndarray:
        # 1. Scale data if required using estimated mean and std
        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m = X.shape[0]

        # 2. Add intercept term to X
        X_with_intercept = np.c_[np.ones(m), X]

        # 3. Concatenate theta_zero and theta
        thetas = np.r_[self.theta_zero, self.theta]

        # 4. Return dot product
        return X_with_intercept.dot(thetas)

    def _score(self, dataset: Dataset) -> float:
        predictions = self.predict(dataset)
        return mse(dataset.y, predictions)