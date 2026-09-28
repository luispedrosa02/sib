import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the Mean Squared Error (MSE) between real and predicted target values.

    Parameters
    ----------
    y_true: np.ndarray
        Real target values.
    y_pred: np.ndarray
        Predicted target values.

    Returns
    -------
    float
        The MSE value.
    """
    return float(np.mean((y_true - y_pred) ** 2))