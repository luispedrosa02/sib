import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Variance Threshold feature selector.
    Removes all features whose variance doesn't meet a threshold.

    Parameters
    ----------
    threshold: float, default=0.0
        Features with a training-set variance lower than or equal to this threshold
        will be removed.
    """

    def __init__(self, threshold: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if threshold < 0:
            raise ValueError("Threshold must be a non-negative value.")
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        mask = self.variance > self.threshold
        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask].tolist() if dataset.features is not None else None
        return Dataset(X=X, y=dataset.y, features=features, label=dataset.label)