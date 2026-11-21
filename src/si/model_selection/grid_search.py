from copy import deepcopy
from itertools import product

import numpy as np

from si.data.dataset import Dataset


def grid_search_cv(model, dataset, hyperparameter_grid, scoring=None,
                   cv=5, seed=None):
    """Compare configurations using the same k validation folds.

    model is an untrained model or a builder function.
    scoring accepts (y_true, y_pred); higher scores are better.
    """
    n_samples = dataset.X.shape[0]

    if dataset.y is None:
        raise ValueError("The dataset must have labels.")
    if not isinstance(cv, (int, np.integer)) or not 2 <= cv <= n_samples:
        raise ValueError("cv must be an integer between 2 and n_samples.")

    # Create the folds once, before trying the configurations.
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n_samples), cv)

    names = list(hyperparameter_grid.keys())
    combinations = product(*hyperparameter_grid.values())
    results = {"hyperparameters": [], "scores": []}

    for values in combinations:
        configuration = dict(zip(names, values))
        fold_scores = []

        for i in range(cv):
            validation_indices = folds[i]
            training_indices = np.concatenate(
                [folds[j] for j in range(cv) if j != i]
            )

            training_dataset = Dataset(
                X=dataset.X[training_indices],
                y=dataset.y[training_indices],
                features=dataset.features,
                label=dataset.label,
            )
            validation_dataset = Dataset(
                X=dataset.X[validation_indices],
                y=dataset.y[validation_indices],
                features=dataset.features,
                label=dataset.label,
            )

            # A different model is used for each configuration and fold.
            if callable(model):
                current_model = model(**configuration)
            else:
                current_model = deepcopy(model)
                for name, value in configuration.items():
                    setattr(current_model, name, value)

            current_model.fit(training_dataset)

            if scoring is None:
                score = current_model.score(validation_dataset)
            else:
                predictions = current_model.predict(validation_dataset)
                score = scoring(validation_dataset.y, predictions)

            fold_scores.append(score)

        results["hyperparameters"].append(configuration)
        results["scores"].append(float(np.mean(fold_scores)))

    if not results["scores"]:
        raise ValueError("The grid must contain at least one configuration.")

    best_index = int(np.argmax(results["scores"]))
    results["best_hyperparameters"] = results["hyperparameters"][best_index]
    results["best_score"] = results["scores"][best_index]

    return results
