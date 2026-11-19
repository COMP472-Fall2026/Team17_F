import numpy as np


def entropy(y):
    if len(y) == 0:     # Return 0 if there are no samples
        return 0.0

    classes, counts = np.unique(y, return_counts=True)      # Count the number of samples belonging to each class
                                                            # Example: y = [0, 0, 0, 1, 1] -> classes = [0, 1], counts = [3, 2]

    probabilities = counts / len(y)        # Calculate the probability of each class: P(class_i) = count_i / N
                                            # P(class_i) = count_i / total samples

    return -np.sum(probabilities * np.log2(probabilities))      # Calculate entropy: H(Y) = -sum(p_i * log2(p_i))
                                                                # Higher entropy indicates greater class uncertainty



# Calculate entropy directly from class counts instead of individual labels
# This avoids recounting labels when evaluating different splits
def entropy_from_counts(counts):

    total = np.sum(counts) # Calculate the total number of samples

    if total == 0:  # Return 0 if there are no samples
        return 0.0

    probabilities = counts[counts > 0] / total          # Calculate class probabilities: P(class_i) = count_i / N
                                                        # Ignore classes with count=0 to avoid log2(0)
                                                        # Example: [7, 3] / 10 = [0.7, 0.3]

    return -np.sum(probabilities * np.log2(probabilities))  # Calculate Entropy: H(Y) = -sum(P(class_i) * log2(P(class_i)))



def gain(y_parent, y_left, y_right):

    parent_entropy = entropy(y_parent)  # Calculate parent node entropy

    # Count the total samples (N) and samples in each child (N_left, N_right)
    n = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)

    if n == 0:      # Return 0 if the parent node is empty
        return 0.0
    
    children_entropy = (                    # Calculate the weighted entropy of the child nodes
        (n_left / n) * entropy(y_left)      # H(children) = (N_left/N)*H(left) + (N_right/N)*H(right)
        + (n_right / n) * entropy(y_right)
    )

    gain = parent_entropy - children_entropy        # Information Gain = H(parent) - H(children)
                                                    # Higher gain means the split reduces more uncertainty
    return gain

#best_split() function finds the feature and threshold that produce the highest information gain.
def best_split(X, y, min_samples_leaf=1):   
    
    # Initialize the best split found so far.
    best_gain = -1
    best_feature = None
    best_threshold = None

    n_samples, n_features = X.shape # Get the number of samples and features.

    classes, parent_counts = np.unique(y, return_counts=True)   # Count the samples belonging to each class

    parent_entropy = entropy_from_counts(parent_counts)     # Calculate parent entropy once for all candidate splits

    # Encode class labels as consecutive indices (0, 1, 2, ...)
    # These indices are used to calculate class counts efficiently
    encoded_y = np.searchsorted(classes, y)

    for feature in range(n_features):   # Evaluate each feature to find the split with the highest information gain

        # Sort the samples by the current feature's values.
        # Apply the same ordering to their class labels.
        values = X[:, feature]
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = encoded_y[sorted_indices]

        # Create a Boolean matrix where each row represents
        # a sample and each column represents a class.
        # True indicates that the sample belongs to that class.
        class_matrix = (
            sorted_labels[:, None] == np.arange(len(classes))
        )

        # Calculate the cumulative number of samples from
        # each class on the left side of every possible split.
        # Subtract these counts from the parent counts to
        # obtain the corresponding right-side class counts.
        left_counts = np.cumsum(class_matrix, axis=0)
        right_counts = parent_counts - left_counts

        # Determine how many samples are assigned to the
        # left and right child nodes at each split position.
        n_left = np.arange(1, n_samples + 1)
        n_right = n_samples - n_left

        left_probs = left_counts / n_left[:, None]  # Calculate class probabilities for the left child at each split

        # Calculate class probabilities for the right child
        # Handle the last position where the right child is empty
        right_probs = np.divide(
            right_counts,
            n_right[:, None],
            out=np.zeros_like(right_counts, dtype=float),
            where=n_right[:, None] != 0
        )

        # Calculate entropy for the left and right children
        # H(S) = -sum(p_i * log2(p_i))
        # Classes with probability 0 do not contribute to entropy
        with np.errstate(divide="ignore", invalid="ignore"):
            left_entropy = -np.sum(
                np.where(
                    left_probs > 0,
                    left_probs * np.log2(left_probs),
                    0
                ),
                axis=1
            )

            right_entropy = -np.sum(
                np.where(
                    right_probs > 0,
                    right_probs * np.log2(right_probs),
                    0
                ),
                axis=1
            )

        # Calculate the weighted entropy of the two child nodes
        # H(children) = (N_left/N)*H(left) + (N_right/N)*H(right)
        weighted_entropy = (
            (n_left / n_samples) * left_entropy
            + (n_right / n_samples) * right_entropy
        )

        # Calculate information gain for every possible split
        # IG = H(parent) - H(children
        gains = parent_entropy - weighted_entropy

        # A valid split must:
        # 1. Separate two different feature values
        # 2. Leave at least min_samples_leaf samples in each child
        valid_splits = (
            (sorted_values[:-1] != sorted_values[1:])
            & (n_left[:-1] >= min_samples_leaf)
            & (n_right[:-1] >= min_samples_leaf)
        )

        # Skip this feature if no valid threshold exists.
        if not np.any(valid_splits):
            continue

        # Exclude invalid split positions by assigning them
        # negative infinity, so they cannot be selected.
        valid_gains = gains[:-1].copy()
        valid_gains[~valid_splits] = -np.inf

        # Find the split position with the highest
        # Information Gain for the current feature.
        best_index = np.argmax(valid_gains)
        current_gain = valid_gains[best_index]

        # Update the overall best split if this feature
        # produces a greater Information Gain.
        if current_gain > best_gain:
            best_gain = current_gain
            best_feature = feature

            # Choose the midpoint between adjacent feature values as the threshold for the decision rule.
            best_threshold = (
                sorted_values[best_index]
                + sorted_values[best_index + 1]
            ) / 2

    # Return the best feature, threshold, and information gain
    return best_feature, best_threshold, best_gain



# Represents a node in the Decision Tree.
# A node can be either:
# 1. Decision node: stores a feature, threshold, and two children.
# 2. Leaf node: stores the predicted class (value).
class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None
    ):
        self.feature = feature  # Feature index used to split the samples
        self.threshold = threshold  # Threshold used to determine the left or right child
        self.left = left    # References to the child nodes
        self.right = right
        self.value = value  # Predicted class label (only assigned to leaf nodes)

    # Check whether the node is a leaf (has a predicted class)
    def is_leaf(self):
        return self.value is not None

    
class DecisionTree:
    
    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        self.max_depth = max_depth  # Maximum depth allowed for the tree
        self.min_samples_split = min_samples_split  # Minimum number of samples required to split a node
        self.min_samples_leaf = min_samples_leaf    # Minimum number of samples required in each child node
        self.root = None    # Root node of the Decision Tree (created during training)


    #Start training
    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0) # Start building the Decision Tree from the root node at depth 0

    def _grow_tree(self, X, y, depth):

        # Count how many samples belong to each class in the current node
        # Example: y=[0,0,1,1,1] -> classes=[0,1], counts=[2,3]
        classes, counts = np.unique(y, return_counts=True)

        # Majority voting: choose the class with the highest frequency
        # Example: counts=[2,3] -> majority_class=1
        majority_class = classes[np.argmax(counts)]

        # Stopping conditions:
        # 1. All samples belong to the same class
        # 2. The maximum depth has been reached
        # 3. There are fewer samples than min_samples_split
        if (
            len(classes) == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            return Node(value=majority_class)

        # Find the feature and threshold with the highest information gain
        feature, threshold, best_gain = best_split(
        X, y, min_samples_leaf=self.min_samples_leaf
        )

        # If no valid split exists or Information Gain <= 0,
        # stop growing and create a leaf using majority voting
        if feature is None or best_gain <= 0:
            return Node(value=majority_class)

        # Divide the samples into left and right groups
        # Left: X[:, feature] <= threshold
        # Right: X[:, feature] > threshold
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Recursively build the left subtree using left side samples
        # depth + 1 means moving one level deeper into the tree
        left_child = self._grow_tree(
            X[left_mask], y[left_mask], depth + 1
        )
        # Recursively build the right subtree using right-side samples
        right_child = self._grow_tree(
            X[right_mask], y[right_mask], depth + 1
        )

        return Node(
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child
        )

    # Predict the class label for each sample
    def predict(self, X):
        predictions = []

        for sample in X:
            # Start at the root and traverse the trained Decision Tree
            # until reaching a leaf node containing the predicted class
            prediction = self._predict_sample(sample, self.root)
            predictions.append(prediction)

        return np.array(predictions)

 
    def _predict_sample(self, sample, node):
        # Base case: if the current node is a leaf,
        # return its stored class label as the prediction
        if node.is_leaf():
            return node.value

        # Check the sample's value for the feature used at this node
        # Example: feature=0, threshold=4.5, sample=[2, 40]
        # Since sample[0]=2 <= 4.5, follow the left child
        if sample[node.feature] <= node.threshold:

            # Recursively continue traversing the left subtree
            return self._predict_sample(sample, node.left)

        else:
            # If the feature value > threshold, follow the right child
            # Continue recursively until a leaf is reached
            return self._predict_sample(sample, node.right)


if __name__ == "__main__":

    # --------------------------------------------------
    # Test 1: Entropy and Information Gain
    # --------------------------------------------------
    print("\n--- Test 1: Entropy and Information Gain ---")

    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    left = np.array([0, 0, 0, 0, 0])
    right = np.array([0, 0, 1, 1, 1])

    counts = np.array([7, 3])

    entropy_value = entropy(labels)
    entropy_counts = entropy_from_counts(counts)
    information_gain = gain(labels, left, right)

    assert np.isclose(entropy_value, 0.8812908992)
    assert np.isclose(entropy_counts, entropy_value)
    assert np.isclose(information_gain, 0.3958156020)

    print("Entropy:", entropy_value)
    print("Entropy from counts:", entropy_counts)
    print("Information gain:", information_gain)
    print("Test passed!")


    # --------------------------------------------------
    # Test 2: Best Split (Single Feature)
    # --------------------------------------------------
    print("\n--- Test 2: Best Split (Single Feature) ---")

    X_test = np.array([
        [0.10],
        [0.20],
        [0.70],
        [0.90]
    ])

    y_test = np.array([0, 0, 1, 1])

    feature, threshold, information_gain = best_split(X_test, y_test)

    assert feature == 0
    assert np.isclose(threshold, 0.45)
    assert np.isclose(information_gain, 1.0)

    print("Best feature:", feature)
    print("Best threshold:", threshold)
    print("Information gain:", information_gain)
    print("Test passed!")


    # --------------------------------------------------
    # Test 3: Best Split (Multiple Features)
    # --------------------------------------------------
    print("\n--- Test 3: Best Split (Multiple Features) ---")

    X_test = np.array([
        [0.1, 0.8],
        [0.2, 0.3],
        [0.3, 0.6],
        [0.7, 0.2],
        [0.8, 0.9],
        [0.9, 0.4]
    ])

    y_test = np.array([0, 0, 0, 1, 1, 1])

    feature, threshold, information_gain = best_split(X_test, y_test)

    assert feature == 0
    assert np.isclose(threshold, 0.5)
    assert np.isclose(information_gain, 1.0)

    print("Best feature:", feature)
    print("Best threshold:", threshold)
    print("Information gain:", information_gain)
    print("Test passed!")


    # --------------------------------------------------
    # Test 4: Mixed Classes and Duplicate Values
    # --------------------------------------------------
    print("\n--- Test 4: Mixed Classes and Duplicate Values ---")

    X_test = np.array([
        [0.1],
        [0.1],
        [0.2],
        [0.8],
        [0.9],
        [0.9]
    ])

    y_test = np.array([0, 0, 1, 1, 1, 0])

    feature, threshold, information_gain = best_split(X_test, y_test)

    assert feature == 0
    assert np.isclose(threshold, 0.15)
    assert np.isclose(information_gain, 0.459147917)

    print("Best feature:", feature)
    print("Best threshold:", threshold)
    print("Information gain:", information_gain)
    print("Test passed!")


    # --------------------------------------------------
    # Test 5: Node Class
    # --------------------------------------------------
    print("\n--- Test 5: Node Class ---")

    decision_node = Node(feature=0, threshold=0.45)
    leaf_node = Node(value=1)

    assert not decision_node.is_leaf()
    assert leaf_node.is_leaf()

    print("Decision node is leaf:", decision_node.is_leaf())
    print("Leaf node is leaf:", leaf_node.is_leaf())
    print("Test passed!")


    # --------------------------------------------------
    # Test 6: Decision Tree Training and Prediction
    # --------------------------------------------------
    print("\n--- Test 6: Training and Prediction ---")

    X_train = np.array([
        [0.10],
        [0.20],
        [0.70],
        [0.90]
    ])

    y_train = np.array([0, 0, 1, 1])

    tree = DecisionTree(max_depth=3)
    tree.fit(X_train, y_train)

    assert tree.root.feature == 0
    assert np.isclose(tree.root.threshold, 0.45)
    assert tree.root.left.value == 0
    assert tree.root.right.value == 1

    new_samples = np.array([
        [0.15],
        [0.80],
        [0.30],
        [0.95]
    ])

    true_labels = np.array([0, 1, 0, 1])
    predictions = tree.predict(new_samples)

    accuracy = np.mean(predictions == true_labels)

    assert np.array_equal(predictions, true_labels)
    assert np.isclose(accuracy, 1.0)

    print("Root feature:", tree.root.feature)
    print("Root threshold:", tree.root.threshold)
    print("Predictions:", predictions)
    print("Accuracy (%):", accuracy * 100)
    print("Test passed!")


    # --------------------------------------------------
    # Test 7: Compare Optimized and Simple Best Split
    # --------------------------------------------------
    print("\n--- Test 7: Randomized Split Comparisons ---")

    # Simple implementation used as a reference
    def simple_best_split(X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])

            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2

                left_y = y[X[:, feature] <= threshold]
                right_y = y[X[:, feature] > threshold]

                current_gain = gain(y, left_y, right_y)

                if current_gain > best_gain:
                    best_gain = current_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain


    rng = np.random.default_rng(42)

    for test_number in range(100):
        X_test = rng.integers(0, 5, size=(20, 4)).astype(float)
        y_test = rng.integers(0, 3, size=20)

        optimized = best_split(X_test, y_test)
        reference = simple_best_split(X_test, y_test)

        assert optimized[0] == reference[0]
        assert np.isclose(optimized[1], reference[1])
        assert np.isclose(optimized[2], reference[2])

    print("All 100 comparison tests passed!")


    # --------------------------------------------------
    # Test 8: Minimum Samples Required to Split
    # --------------------------------------------------
    print("\n--- Test 8: min_samples_split ---")

    X_test = np.array([
        [0.1],
        [0.2],
        [0.8],
        [0.9]
    ])

    y_test = np.array([0, 0, 1, 1])

    # The node has 4 samples but requires at least 5 to split
    tree = DecisionTree(max_depth=5, min_samples_split=5)
    tree.fit(X_test, y_test)

    assert tree.root.is_leaf()

    print("min_samples_split test passed!")


    # --------------------------------------------------
    # Test 9: Minimum Samples Required in Each Leaf
    # --------------------------------------------------
    print("\n--- Test 9: min_samples_leaf ---")

    # Four samples cannot be split into two groups of at least 3
    tree = DecisionTree(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=3
    )

    tree.fit(X_test, y_test)

    assert tree.root.is_leaf()

    print("min_samples_leaf test passed!")


    # --------------------------------------------------
    # All Tests Completed
    # --------------------------------------------------
    print("\nAll Decision Tree tests passed!")
