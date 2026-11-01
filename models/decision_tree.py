
import numpy as np


def gini_impurity(y):
    if len(y) == 0:
        return 0.0

    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    gini = 1 - np.sum(probabilities ** 2)

    return gini


def entropy(y):
    # If the node contains no samples, return 0
    if len(y) == 0:
        return 0.0

    # Count the number of samples belonging to each class
    # Example: y = [0, 0, 0, 1, 1] -> classes = [0, 1], counts = [3, 2]
    classes, counts = np.unique(y, return_counts=True)

    # Calculate the probability of each class: P(class_i) = count_i / N
    # Example: probabilities = [3/5, 2/5] = [0.6, 0.4]
    probabilities = counts / len(y)

    # Calculate Entropy: H(Y) = -sum(P(class_i) * log2(P(class_i)))
    # Example: H(Y) = -(0.6*log2(0.6) + 0.4*log2(0.4)) = 0.971
    return -np.sum(probabilities * np.log2(probabilities))



# Calculate entropy directly from class counts instead of individual labels
# This avoids recounting labels when evaluating different splits
def entropy_from_counts(counts):

    # Calculate the total number of samples: N = sum(counts)
    # Example: counts = [7, 3] -> total = 10
    total = np.sum(counts)

    # If there are no samples, return 0 to avoid division by zero
    if total == 0:
        return 0.0

    # Calculate class probabilities: P(class_i) = count_i / N
    # Ignore classes with count=0 to avoid log2(0)
    # Example: [7, 3] / 10 = [0.7, 0.3]
    probabilities = counts[counts > 0] / total

    # Calculate Entropy: H(Y) = -sum(P(class_i) * log2(P(class_i)))
    # Example: -(0.7*log2(0.7) + 0.3*log2(0.3)) = 0.8813
    return -np.sum(probabilities * np.log2(probabilities))




def gain(y_parent, y_left, y_right):
    # Calculate parent entropy
    parent_entropy = entropy(y_parent)

    # Count the total samples (N) and samples in each child (N_left, N_right)
    n = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)

    # Avoid division by zero if the parent node is empty
    if n == 0:
        return 0.0

    # Calculate the weighted entropy after splitting:
    # H(children) = (N_left/N)*H(Y_left) + (N_right/N)*H(Y_right)
    # Each child's entropy is weighted by its proportion of samples
    children_entropy = (
        (n_left / n) * entropy(y_left)
        + (n_right / n) * entropy(y_right)
    )


    # Information Gain = H(parent) - H(children)
    # Higher gain means the split reduces more uncertainty
    gain = parent_entropy - children_entropy

    return gain


# Try different features and thresholds to find the split
# that maximizes Information Gain (reduces entropy the most).

def best_split(X, y):
    # Initialize the best split found so far
    best_gain = -1
    best_feature = None
    best_threshold = None

    # Get the number of samples (rows) and features (columns)
    n_samples, n_features = X.shape

    # Find all unique class labels and count their occurrences
    # Example: y = [0, 0, 0, 1, 1]
    # classes = [0, 1], parent_counts = [3, 2]
    classes, parent_counts = np.unique(y, return_counts=True)

    # Calculate parent entropy only once
    # H(parent) = -sum(p_i * log2(p_i))
    parent_entropy = entropy_from_counts(parent_counts)

    # Convert class labels to indices: 0, 1, 2, ...
    # This also works when original labels are not consecutive
    encoded_y = np.searchsorted(classes, y)

    # Try every feature (column of X)
    for feature in range(n_features):

        # Extract all values for the current feature
        values = X[:, feature]

        # Sort feature values and keep their class labels aligned
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = encoded_y[sorted_indices]

        # Initially, all samples are on the right
        left_counts = np.zeros(len(classes), dtype=int)
        right_counts = parent_counts.copy()

        # Move samples one by one from right to left
        for i in range(n_samples - 1):

            # Get the class of the current sample
            label = sorted_labels[i]

            # Move this sample from right to left
            left_counts[label] += 1
            right_counts[label] -= 1

            # Skip if the next feature value is identical
            # We cannot place a threshold between equal values
            if sorted_values[i] == sorted_values[i + 1]:
                continue

            # Number of samples on each side
            n_left = i + 1
            n_right = n_samples - n_left

            # Calculate entropy of the two groups
            left_entropy = entropy_from_counts(left_counts)
            right_entropy = entropy_from_counts(right_counts)

            # Weighted entropy after splitting
            # H(children) = (n_left/n)*H(left)
            #             + (n_right/n)*H(right)
            children_entropy = (
                (n_left / n_samples) * left_entropy
                + (n_right / n_samples) * right_entropy
            )

            # Information Gain = parent entropy - children entropy
            current_gain = parent_entropy - children_entropy

            # Keep the split with the highest Information Gain
            if current_gain > best_gain:
                best_gain = current_gain
                best_feature = feature

                # Threshold is the midpoint of adjacent values
                best_threshold = (
                    sorted_values[i] + sorted_values[i + 1]
                ) / 2

    # Return the best feature, threshold, and Information Gain
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
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    # Check whether the node is a leaf (has a predicted class)
    def is_leaf(self):
        return self.value is not None

    
class DecisionTree:
    

    def __init__(self, max_depth=5):     
        # Maximum depth allowed for the tree to prevent overfitting
        # Example: max_depth=5 means the tree can split up to depth 5 --> hyperparameter
        self.max_depth = max_depth
        self.root = None

    #Start training
    def fit(self, X, y):
        # Train the tree using features X and class labels y
        # Start building from the root at depth 0
        self.root = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):

        # Count how many samples belong to each class in the current node
        # Example: y=[0,0,1,1,1] -> classes=[0,1], counts=[2,3]
        classes, counts = np.unique(y, return_counts=True)

        # Majority voting: choose the class with the highest frequency
        # Example: counts=[2,3] -> majority_class=1
        majority_class = classes[np.argmax(counts)]

        # Stopping conditions:
        # 1. Node is pure (all samples belong to the same class)
        # 2. Maximum tree depth has been reached
        # In either case, create a leaf predicting the majority class
        if len(classes) == 1 or depth >= self.max_depth:
            return Node(value=majority_class)

        # Find the feature and threshold with maximum Information Gain
        # IG = H(parent) - weighted H(children)
        feature, threshold, best_gain = best_split(X, y)

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






# Test our Gini impurity function
if __name__ == "__main__":
    labels = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    print("Entropy:", entropy(labels))
    
    left = np.array([0, 0, 0, 0, 0])
    right = np.array([0, 0, 1, 1, 1])

    print("Gain:", gain(labels, left, right))

    print("Gini impurity:", gini_impurity(labels))


    X_test = np.array([
        [0.10],
        [0.20],
        [0.70],
        [0.90]
    ])

    y_test = np.array([0, 0, 1, 1])

    feature, threshold, score = best_split(X_test, y_test)

    print("Best feature:", feature)
    print("Best threshold:", threshold)
    print("Best gain:", score)

    decision_node = Node(feature=0, threshold=0.45)
    leaf_node = Node(value=1)

    print("Decision node is leaf:", decision_node.is_leaf())
    print("Leaf node is leaf:", leaf_node.is_leaf())

    tree = DecisionTree(max_depth=3)
    tree.fit(X_test, y_test)

    print("Root feature:", tree.root.feature)
    print("Root threshold:", tree.root.threshold)
    print("Left prediction:", tree.root.left.value)
    print("Right prediction:", tree.root.right.value)


    new_samples = np.array([
        [0.15],
        [0.80],
        [0.30],
        [0.95]
    ])

    predictions = tree.predict(new_samples)

    print("Predictions:", predictions)


    # Correct labels for our four new samples
    true_labels = np.array([0, 1, 0, 1])

    # Compare predictions with correct labels
    accuracy = np.mean(predictions == true_labels)

    print("Accuracy:", accuracy)
    print("Accuracy (%):", accuracy * 100)

    counts = np.array([7, 3])

    print("Entropy from counts:", entropy_from_counts(counts))


    # Test sorting feature values while keeping labels aligned
    X_example = np.array([[6], [2], [8], [1], [4]])
    y_example = np.array([1, 0, 1, 0, 0])

    # Get the indices that would sort the first feature
    sorted_indices = np.argsort(X_example[:, 0])

    # Sort feature values and their corresponding labels
    sorted_values = X_example[sorted_indices, 0]
    sorted_labels = y_example[sorted_indices]

    print("Sorted indices:", sorted_indices)
    print("Sorted values:", sorted_values)
    print("Sorted labels:", sorted_labels)

    
    # Count samples belonging to each class
    classes, counts = np.unique(
        sorted_labels, return_counts=True
    )

    # Initially, no samples are on the left
    left_counts = np.zeros(len(classes), dtype=int)

    # All samples start on the right
    right_counts = counts.copy()

    print("Before moving:")
    print("Left counts:", left_counts)
    print("Right counts:", right_counts)

    # Get the class label of the first sorted sample
    label = sorted_labels[0]

    # Move one sample from right to left
    left_counts[label] += 1
    right_counts[label] -= 1

    print("After moving:")
    print("Left counts:", left_counts)
    print("Right counts:", right_counts)







