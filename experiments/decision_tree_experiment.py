import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.decision_tree import DecisionTree
import time

# Load the image and sentiment datasets
image_data = pd.read_csv("data/image_train.csv")
sentiment_data = pd.read_csv("data/sentiment_train.csv")

# Separate features and class labels
X_image = image_data.drop(columns=["id", "label"]).to_numpy()
y_image = image_data["label"].to_numpy()

X_sentiment = sentiment_data.drop(columns=["id", "label"]).to_numpy()
y_sentiment = sentiment_data["label"].to_numpy()

# Split each dataset into 80% training and 20% validation
X_image_train, X_image_val, y_image_train, y_image_val = train_test_split(
    X_image,
    y_image,
    test_size=0.2,
    random_state=42,
    stratify=y_image
)

X_sent_train, X_sent_val, y_sent_train, y_sent_val = train_test_split(
    X_sentiment,
    y_sentiment,
    test_size=0.2,
    random_state=42,
    stratify=y_sentiment
)

# Display the training and validation dimensions
print("Image training:", X_image_train.shape)
print("Image validation:", X_image_val.shape)
print("Sentiment training:", X_sent_train.shape)
print("Sentiment validation:", X_sent_val.shape)


# Experiment 1: Effect of maximum depth on image classification
def experiment_max_depth():
    depths = [5, 10, 15, 20, 25, 30]
    image_results = []

    for depth in depths:
        print(f"\nTraining Decision Tree with max_depth={depth}")

        tree = DecisionTree(max_depth=depth)

        # Measure training time
        start_time = time.perf_counter()
        tree.fit(X_image_train, y_image_train)
        training_time = time.perf_counter() - start_time

        # Calculate training accuracy
        train_predictions = tree.predict(X_image_train)
        train_accuracy = np.mean(train_predictions == y_image_train)

        # Calculate validation accuracy
        val_predictions = tree.predict(X_image_val)
        val_accuracy = np.mean(val_predictions == y_image_val)

        # Store the results
        image_results.append({
            "max_depth": depth,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time
        })

        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print(f"Training time: {training_time:.2f} seconds")

    # Display the results in a table
    image_results_df = pd.DataFrame(image_results)

    print("\nImage Decision Tree Depth Comparison:")
    print(image_results_df.to_string(index=False))




# Experiment 2: Effect of training-set size on image classification
def experiment_training_size():
    training_sizes = [1000, 2000, 4000, 8000, 16000]
    size_results = []

    for size in training_sizes:

        # Select a stratified subset of the training data
        if size == len(X_image_train):
            X_subset = X_image_train
            y_subset = y_image_train
        else:
            X_subset, _, y_subset, _ = train_test_split(
                X_image_train,
                y_image_train,
                train_size=size,
                random_state=42,
                stratify=y_image_train
            )

        # Keep the maximum depth fixed
        tree = DecisionTree(max_depth=10)

        # Measure training time
        start_time = time.perf_counter()
        tree.fit(X_subset, y_subset)
        training_time = time.perf_counter() - start_time

        # Calculate training accuracy
        train_predictions = tree.predict(X_subset)
        train_accuracy = np.mean(train_predictions == y_subset)

        # Calculate validation accuracy
        val_predictions = tree.predict(X_image_val)
        val_accuracy = np.mean(val_predictions == y_image_val)

        # Store the results
        size_results.append({
            "training_size": size,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time
        })

        print(f"\nTraining samples: {size}")
        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print(f"Training time: {training_time:.2f} seconds")

    # Display the results in a table
    size_results_df = pd.DataFrame(size_results)

    print("\nImage Decision Tree Training Size Comparison:")
    print(size_results_df.to_string(index=False))



# Experiment 3: Effect of min_samples_split on image classification
def experiment_min_samples_split():
    min_samples_split_values = [2, 5, 10, 20, 50]
    split_results = []

    for min_split in min_samples_split_values:
        print(f"\nTraining Decision Tree with min_samples_split={min_split}")

        # Create a tree with fixed depth and minimum leaf size
        tree = DecisionTree(
            max_depth=20,
            min_samples_split=min_split,
            min_samples_leaf=1
        )

        # Measure training time
        start_time = time.perf_counter()
        tree.fit(X_image_train, y_image_train)
        training_time = time.perf_counter() - start_time

        # Calculate training accuracy
        train_predictions = tree.predict(X_image_train)
        train_accuracy = np.mean(train_predictions == y_image_train)

        # Calculate validation accuracy
        val_predictions = tree.predict(X_image_val)
        val_accuracy = np.mean(val_predictions == y_image_val)

        # Store the results
        split_results.append({
            "min_samples_split": min_split,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time
        })

        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print(f"Training time: {training_time:.2f} seconds")

    # Display the results in a table
    split_results_df = pd.DataFrame(split_results)

    print("\nImage Decision Tree min_samples_split Comparison:")
    print(split_results_df.to_string(index=False))



# Experiment 4: Effect of min_samples_leaf on image classification
def experiment_min_samples_leaf():
    min_samples_leaf_values = [1, 2, 5, 10, 20]
    leaf_results = []

    for min_leaf in min_samples_leaf_values:
        print(f"\nTraining Decision Tree with min_samples_leaf={min_leaf}")

        # Create a tree with fixed depth and minimum split size
        tree = DecisionTree(
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=min_leaf
        )

        # Measure training time
        start_time = time.perf_counter()
        tree.fit(X_image_train, y_image_train)
        training_time = time.perf_counter() - start_time

        # Calculate training accuracy
        train_predictions = tree.predict(X_image_train)
        train_accuracy = np.mean(train_predictions == y_image_train)

        # Calculate validation accuracy
        val_predictions = tree.predict(X_image_val)
        val_accuracy = np.mean(val_predictions == y_image_val)

        # Store the results
        leaf_results.append({
            "min_samples_leaf": min_leaf,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time
        })

        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print(f"Training time: {training_time:.2f} seconds")

    # Display the results in a table
    leaf_results_df = pd.DataFrame(leaf_results)

    print("\nImage Decision Tree min_samples_leaf Comparison:")
    print(leaf_results_df.to_string(index=False))




# Decision Tree for sentiment classification
def experiment_sentiment():

    X_sent_small = X_sent_train
    y_sent_small = y_sent_train

    # Create a Decision Tree with maximum depth 5
    sent_tree = DecisionTree(max_depth=5)

    # Measure training time
    start_time = time.perf_counter()
    sent_tree.fit(X_sent_small, y_sent_small)
    training_time = time.perf_counter() - start_time

    # Calculate training accuracy
    train_predictions = sent_tree.predict(X_sent_small)
    train_accuracy = np.mean(train_predictions == y_sent_small)

    # Calculate validation accuracy
    val_predictions = sent_tree.predict(X_sent_val)
    val_accuracy = np.mean(val_predictions == y_sent_val)

    # Display the results
    print("\nSentiment Decision Tree Results:")
    print("Training samples:", len(X_sent_small))
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation accuracy: {val_accuracy:.2%}")
    print(f"Training time: {training_time:.2f} seconds")


# Experiment 6: Effect of maximum depth on sentiment classification
def experiment_sentiment_depth():
    depths = [5, 10, 15, 20]
    sentiment_results = []

    for depth in depths:
        print(f"\nTraining Sentiment Decision Tree with max_depth={depth}")

        # Create a Decision Tree with the current maximum depth
        tree = DecisionTree(max_depth=depth)

        # Measure training time
        start_time = time.perf_counter()
        tree.fit(X_sent_train, y_sent_train)
        training_time = time.perf_counter() - start_time

        # Calculate training accuracy
        train_predictions = tree.predict(X_sent_train)
        train_accuracy = np.mean(train_predictions == y_sent_train)

        # Calculate validation accuracy
        val_predictions = tree.predict(X_sent_val)
        val_accuracy = np.mean(val_predictions == y_sent_val)

        # Store the results
        sentiment_results.append({
            "max_depth": depth,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time
        })

        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print(f"Training time: {training_time:.2f} seconds")

    # Display the results in a table
    sentiment_results_df = pd.DataFrame(sentiment_results)

    print("\nSentiment Decision Tree Depth Comparison:")
    print(sentiment_results_df.to_string(index=False))


# Experiment 7: Effect of min_samples_leaf on sentiment classification
def experiment_sentiment_min_samples_leaf():
    min_samples_leaf_values = [1, 2, 5, 10, 20]
    leaf_results = []

    for min_leaf in min_samples_leaf_values:
        print(f"\nTraining Sentiment Decision Tree with min_samples_leaf={min_leaf}")

        # Create a tree with fixed depth and minimum split size
        tree = DecisionTree(
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=min_leaf
        )

        # Measure training time
        start_time = time.perf_counter()
        tree.fit(X_sent_train, y_sent_train)
        training_time = time.perf_counter() - start_time

        # Calculate training accuracy
        train_predictions = tree.predict(X_sent_train)
        train_accuracy = np.mean(train_predictions == y_sent_train)

        # Calculate validation accuracy
        val_predictions = tree.predict(X_sent_val)
        val_accuracy = np.mean(val_predictions == y_sent_val)

        # Store the results
        leaf_results.append({
            "min_samples_leaf": min_leaf,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time
        })

        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print(f"Training time: {training_time:.2f} seconds")

    # Display the results in a table
    leaf_results_df = pd.DataFrame(leaf_results)

    print("\nSentiment Decision Tree min_samples_leaf Comparison:")
    print(leaf_results_df.to_string(index=False))


# Experiment 8: Effect of min_samples_split on sentiment classification
def experiment_sentiment_min_samples_split():
    min_samples_split_values = [2, 5, 10, 20, 50]
    split_results = []

    for min_split in min_samples_split_values:
        print(f"\nTraining Sentiment Decision Tree with min_samples_split={min_split}")

        # Create a tree with fixed depth and minimum leaf size
        tree = DecisionTree(
            max_depth=5,
            min_samples_split=min_split,
            min_samples_leaf=1
        )

        # Measure training time
        start_time = time.perf_counter()
        tree.fit(X_sent_train, y_sent_train)
        training_time = time.perf_counter() - start_time

        # Calculate training accuracy
        train_predictions = tree.predict(X_sent_train)
        train_accuracy = np.mean(train_predictions == y_sent_train)

        # Calculate validation accuracy
        val_predictions = tree.predict(X_sent_val)
        val_accuracy = np.mean(val_predictions == y_sent_val)

        # Store the results
        split_results.append({
            "min_samples_split": min_split,
            "training_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
            "training_time": training_time
        })

        print(f"Training accuracy: {train_accuracy:.2%}")
        print(f"Validation accuracy: {val_accuracy:.2%}")
        print(f"Training time: {training_time:.2f} seconds")

    # Display the results in a table
    split_results_df = pd.DataFrame(split_results)

    print("\nSentiment Decision Tree min_samples_split Comparison:")
    print(split_results_df.to_string(index=False))



# Select which experiment to run
if __name__ == "__main__":

    experiment = 8

    if experiment == 1:
        experiment_max_depth()

    elif experiment == 2:
        experiment_training_size()

    elif experiment == 3:
        experiment_min_samples_split()

    elif experiment == 4:
        experiment_min_samples_leaf()

    elif experiment == 5:
        experiment_sentiment()
    
    elif experiment == 6:
        experiment_sentiment_depth()

    elif experiment == 7:
        experiment_sentiment_min_samples_leaf()

    elif experiment == 8:
        experiment_sentiment_min_samples_split()

    else:
        print("Invalid experiment number.")



