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
        
        # TODO: implement this function
        # Step 1: Shuffle the indices. (use np.random.shuffle function)
        # ...

        # Step 2: Calculate the size for each fold split
        # fold_size = ...
        folds = []

        for i in range(n_splits):
            # Step 3: Split the shuffled indices into n_splits folds.
            test_indices = total_indices[i * fold_size:(i + 1) * fold_size]
            # train_indices = ...
            folds.append((train_indices, test_indices))

        return folds

    def fit(self, X, y):
        """
        Perform nested cross-validation to evaluate model performance.

        :param X: Feature matrix.
        :param y: Target vector.
        :return: List of outer fold accuracies and mean accuracy.
        """
        # TODO: implement this function
        # Step 1: Create outer folds using _create_folds().
        # outer_folds = ...
        # Step 2: Initialize a list to store outer fold results.
        outer_results = []

        # Step 3: Loop through each outer fold.
        for outer_idx, (train_idx, test_idx) in enumerate(outer_folds):
            # Step 4: Split data into outer train/test sets.
            # ...

            # Inner loop for hyperparameter tuning
            # Step 5: Initialize best parameters and best score.
            best_params = None
            best_score = -np.inf

            # Step 6: Create inner folds from outer training data.
            # inner_folds = ...

            # Step 7: Iterate over hyperparameter combinations.
            for param_combination in product(*self.param_grid.values()):
                params = dict(zip(self.param_grid.keys(), param_combination))
                inner_scores = []

                # Step 8: Train model using inner train sets and evaluate using validation sets.
                for inner_train_idx, inner_val_idx in inner_folds:
                    # Split data into inner train/test sets
                    # ...

                    model = self.model_class(**params)
                    # Train model with the current parameters
                    # ...

                    # Evaluate on validation set and update inner_scores
                    # ...

                # Step 9: Select the best hyperparameter combination.
                # Compute mean score and update best parameters
                # ...

            # Step 10: Train the final model on outer training data using best parameters.
            # final_model = ...
            # ...

            # Step 11: Evaluate the final model on outer test set and store the result.
            # ...
            # outer_accuracy = ...
            # ...

            print(f"Outer Fold {outer_idx + 1} - Best Params: {best_params}, Accuracy: {outer_accuracy:.4f}")

        mean_outer_accuracy = np.mean(outer_results)
        print(f"\nOverall Accuracy from Nested Cross-Validation: {mean_outer_accuracy:.4f}")
        # Step 12: Return list of outer fold accuracies and overall mean accuracy.
        return outer_results, mean_outer_accuracy