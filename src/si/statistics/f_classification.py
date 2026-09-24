from typing import Tuple
import numpy as np
from scipy import stats
from si.data.dataset import Dataset


def f_classification(dataset: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the ANOVA F-value for each feature across classes.

    Parameters
    ----------
    dataset: Dataset
        A labeled dataset.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        F-values and p-values for each feature.
    """
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    f_values, p_values = stats.f_oneway(*groups)
    return np.nan_to_num(f_values), np.nan_to_num(p_values)