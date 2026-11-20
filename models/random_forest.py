import numpy as np

from models.decision_tree import DecisionTree, Node, best_split


class RandomForestTree(DecisionTree):
    def __init__(
        self,
        max_depth=8,
        max_features=None,
        random_state=None,
    ):
        super().__init__(max_depth=max_depth)
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        # Set the random generator used to choose features
        self.rng = np.random.default_rng(self.random_state)
        return super().fit(X, y)

    def _grow_tree(self, X, y, depth):
        # Count the labels and find the most common one
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]

        # Stop if all labels agree or the maximum depth is reached
        if len(classes) == 1 or depth >= self.max_depth:
            return Node(value=majority_class)

        if self.max_features is None:
            # Consider all features when no limit is given
            feature, threshold, best_gain = best_split(X, y)
        else:
            # Choose random features for this split without repeats
            feature_indices = self.rng.choice(
                X.shape[1],
                size=min(self.max_features, X.shape[1]),
                replace=False,
            )

            # Find the best split using only the selected features
            local_feature, threshold, best_gain = best_split(
                X[:, feature_indices], y
            )

            # Find the selected feature's original column number
            feature = None
            if local_feature is not None:
                feature = int(feature_indices[local_feature])

        # Use the most common label if no useful split is found
        if feature is None or best_gain <= 0:
            return Node(value=majority_class)

        # Separate examples using the selected feature and threshold
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Both sides must contain examples
        if not np.any(left_mask) or not np.any(right_mask):
            return Node(value=majority_class)

        # Repeat the process for the left and right groups
        left_child = self._grow_tree(
            X[left_mask], y[left_mask], depth + 1
        )
        right_child = self._grow_tree(
            X[right_mask], y[right_mask], depth + 1
        )

        # Store this split and its two branches
        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )


class RandomForest:
    def __init__(
        self,
        n_trees=10,
        max_depth=16,
        max_features="sqrt",
        random_state=42,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        # Convert the training data into NumPy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Check that the training examples contain valid features
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must be a nonempty table of features.")

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("Each training example must have one label.")

        if (
            not np.issubdtype(X.dtype, np.number)
            or not np.isfinite(X).all()
        ):
            raise ValueError("X must contain only finite numbers.")

        if (
            not isinstance(self.n_trees, (int, np.integer))
            or self.n_trees < 1
        ):
            raise ValueError("n_trees must be a positive integer.")

        # Calculate how many features each split can consider
        feature_count = self._get_feature_count(X.shape[1])
        rng = np.random.default_rng(self.random_state)

        # Remember the feature count and possible labels
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)

        # Start a new forest when fit is called
        self.trees = []

        for _ in range(self.n_trees):
            # Give each tree a sample of training rows, allowing repeats
            X_sample, y_sample = self._bootstrap_sample(X, y, rng)

            # Give each tree its own repeatable random seed
            tree_seed = int(rng.integers(0, 2**31 - 1))

            tree = RandomForestTree(
                max_depth=self.max_depth,
                max_features=feature_count,
                random_state=tree_seed,
            )

            # Train the tree and add it to the forest
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

        return self

    def predict(self, X):
        # Check that the forest has been trained
        if not self.trees:
            raise ValueError("Train the forest with fit before predicting.")

        X = np.asarray(X)

        # New examples must have the same feature columns as training
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X must have the same feature columns as training."
            )

        if (
            not np.issubdtype(X.dtype, np.number)
            or not np.isfinite(X).all()
        ):
            raise ValueError("X must contain only finite numbers.")

        # Create one row per example and one column per possible label
        votes = np.zeros((len(X), len(self.classes_)), dtype=int)
        rows = np.arange(len(X))

        # Count one vote from each tree for every example
        for tree in self.trees:
            predictions = tree.predict(X)
            label_columns = np.searchsorted(self.classes_, predictions)
            votes[rows, label_columns] += 1

        # Select the label with the most votes
        # Ties go to the smallest label
        winning_columns = np.argmax(votes, axis=1)
        return self.classes_[winning_columns]

    def _bootstrap_sample(self, X, y, rng):
        # Draw the same number of rows as the original dataset
        # A row can be selected more than once
        n_samples = len(X)
        indices = rng.integers(0, n_samples, size=n_samples)

        return X[indices], y[indices]

    def _get_feature_count(self, n_features):
        # Use the square root of the feature count, rounded down
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))

        # None means every feature can be considered.
        if self.max_features is None:
            return n_features

        # A fixed count cannot exceed the available number of features
        if isinstance(self.max_features, int) and self.max_features > 0:
            return min(self.max_features, n_features)

        raise ValueError(
            "invalid max_features "
        )
