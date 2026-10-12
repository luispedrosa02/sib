import numpy as np

def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """Aplica a função sigmoide aos valores de entrada."""
    return 1 / (1 + np.exp(-X))