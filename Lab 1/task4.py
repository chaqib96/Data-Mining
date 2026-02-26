import numpy as np
from sklearn.metrics import accuracy_score
from itertools import product

class NestedCrossValidator:
    def __init__(self, model_class, param_grid, outer_splits=5, inner_splits=3, random_seed=None):
        """
        Initialize the nested cross-validator.

        :param model_class: The model class to be used (e.g., SVC, RandomForestClassifier).
        :param param_grid: Dictionary of hyperparameters and their values to search.
        :param outer_splits: Number of splits for the outer cross-validation.
        :param inner_splits: Number of splits for the inner cross-validation.
        :param random_seed: Random seed for reproducibility.
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.random_seed = random_seed

    def _create_folds(self, X, y, n_splits):
        """
        Manually create folds for cross-validation.

        :param X: Feature matrix.
        :param y: Target vector.
        :param n_splits: Number of splits for cross-validation.
        :return: List of (train_indices, test_indices) for each fold.
        """
        np.random.seed(self.random_seed) # DO NOT DELETE THIS LINE
        total_indices = np.arange(len(X))
        np.random.shuffle(total_indices)
        n_samples = len(total_indices)
        base_size = n_samples // n_splits
        remainder = n_samples % n_splits
        # Distribute remainder so each sample is in exactly one test set
        fold_sizes = [base_size + (1 if i < remainder else 0) for i in range(n_splits)]
        folds = []
        start = 0
        for i in range(n_splits):
            end = start + fold_sizes[i]
            test_indices = total_indices[start:end]
            train_indices = np.concatenate([
                total_indices[:start],
                total_indices[end:]
            ])
            folds.append((train_indices, test_indices))
            start = end
        return folds

    def fit(self, X, y):
        """
        Perform nested cross-validation to evaluate model performance.

        :param X: Feature matrix.
        :param y: Target vector.
        :return: List of outer fold accuracies and mean accuracy.
        """
        outer_folds = self._create_folds(X, y, self.outer_splits)
        outer_results = []

        for outer_idx, (train_idx, test_idx) in enumerate(outer_folds):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            best_params = None
            best_score = -np.inf
            inner_folds = self._create_folds(X_train, y_train, self.inner_splits)

            for param_combination in product(*self.param_grid.values()):
                params = dict(zip(self.param_grid.keys(), param_combination))
                inner_scores = []
                for inner_train_idx, inner_val_idx in inner_folds:
                    X_inner_train = X_train[inner_train_idx]
                    y_inner_train = y_train[inner_train_idx]
                    X_inner_val = X_train[inner_val_idx]
                    y_inner_val = y_train[inner_val_idx]
                    model = self.model_class(**params)
                    model.fit(X_inner_train, y_inner_train)
                    score = accuracy_score(y_inner_val, model.predict(X_inner_val))
                    inner_scores.append(score)
                mean_score = np.mean(inner_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

            final_model = self.model_class(**best_params)
            final_model.fit(X_train, y_train)
            outer_accuracy = accuracy_score(y_test, final_model.predict(X_test))
            outer_results.append(outer_accuracy)
            print(f"Outer Fold {outer_idx + 1} - Best Params: {best_params}, Accuracy: {outer_accuracy:.4f}")

        mean_outer_accuracy = np.mean(outer_results)
        print(f"\nOverall Accuracy from Nested Cross-Validation: {mean_outer_accuracy:.4f}")
        # Step 12: Return list of outer fold accuracies and overall mean accuracy.
        return outer_results, mean_outer_accuracy