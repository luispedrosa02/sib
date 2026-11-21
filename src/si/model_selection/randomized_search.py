import numpy as np

from copy import deepcopy
from si.data.dataset import Dataset


def randomized_search_cv(model, dataset, hyperparameter_distributions,
                         scoring=None, cv=5, n_iter=10, seed=None):
    n_samples = dataset.X.shape[0]

    if dataset.y is None:
        raise ValueError("The dataset must have labels.")
    if not isinstance(cv, (int, np.integer)) or not 2 <= cv <= n_samples:
        raise ValueError("cv must be an integer between 2 and n_samples.")
    if not isinstance(n_iter, (int, np.integer)) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer.")

    # Criar os folds uma única vez
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n_samples), cv)

    results = {"hyperparameters": [], "scores": []}

    for _ in range(n_iter):
        configuration = {}

        # Sortear um valor para cada hiperparâmetro
        for name, distribution in hyperparameter_distributions.items():
            if callable(distribution):
                configuration[name] = distribution(rng)
            else:
                if len(distribution) == 0:
                    raise ValueError(f"The list for {name} cannot be empty.")

                index = int(rng.integers(len(distribution)))
                configuration[name] = distribution[index]

        fold_scores = []

        for i in range(cv):
            validation_indices = folds[i]
            training_indices = np.concatenate(
                [folds[j] for j in range(cv) if j != i]
            )

            training_dataset = Dataset(
                dataset.X[training_indices],
                dataset.y[training_indices],
                features=dataset.features,
                label=dataset.label
            )
            validation_dataset = Dataset(
                dataset.X[validation_indices],
                dataset.y[validation_indices],
                features=dataset.features,
                label=dataset.label
            )

            # Criar um modelo separado para cada fold
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

    best_index = int(np.argmax(results["scores"]))

    results["best_hyperparameters"] = results["hyperparameters"][best_index]
    results["best_score"] = results["scores"][best_index]

    return results
    
def log_uniform(low, high):
    if not 0 < low < high:
        raise ValueError("Os limites devem respeitar 0 < low < high.")

    def sample(rng):
        exponent = rng.uniform(np.log10(low), np.log10(high))
        return float(10.0 ** exponent)

    return sample

