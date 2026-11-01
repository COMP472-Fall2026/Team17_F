import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.decision_tree import DecisionTree

# Load the training datasets
image_data = pd.read_csv("data/image_train.csv")
sentiment_data = pd.read_csv("data/sentiment_train.csv")

# Check the number of rows and columns
print("Image dataset shape:", image_data.shape)
print("Sentiment dataset shape:", sentiment_data.shape)

# Show the first five rows of each dataset
print("\nImage data:")
print(image_data.head())

print("\nSentiment data:")
print(sentiment_data.head())


# Separate features and labels

# Image dataset
X_image = image_data.drop(columns=["id", "label"]).to_numpy()
y_image = image_data["label"].to_numpy()

# Sentiment dataset
X_sentiment = sentiment_data.drop(columns=["id", "label"]).to_numpy()
y_sentiment = sentiment_data["label"].to_numpy()

# Verify dimensions
print("\nImage features:", X_image.shape)
print("Image labels:", y_image.shape)

print("\nSentiment features:", X_sentiment.shape)
print("Sentiment labels:", y_sentiment.shape)


# Split image dataset into training and validation (80 traing/ 20 validation)
X_image_train, X_image_val, y_image_train, y_image_val = train_test_split(
    X_image,
    y_image,
    test_size=0.2,
    random_state=42,
    stratify=y_image
)

# Split sentiment dataset into training and validation (80 traing/ 20 validation)
X_sent_train, X_sent_val, y_sent_train, y_sent_val = train_test_split(
    X_sentiment,
    y_sentiment,
    test_size=0.2,
    random_state=42,
    stratify=y_sentiment
)

# Verify the splits
print("\nImage training:", X_image_train.shape)
print("Image validation:", X_image_val.shape)

print("\nSentiment training:", X_sent_train.shape)
print("Sentiment validation:", X_sent_val.shape)

# Start with a small subset to test our Decision Tree
X_small = X_image_train[:1000]
y_small = y_image_train[:1000]

# Create a Decision Tree with maximum depth 3
tree = DecisionTree(max_depth=8)

# Train the Decision Tree
tree.fit(X_small, y_small)

# Predict labels for the same 200 samples
predictions = tree.predict(X_small)

# Calculate training accuracy
accuracy = np.mean(predictions == y_small)

print("Training accuracy:", accuracy)


# Evaluate the trained tree on unseen validation samples
# These samples were not used to train the Decision Tree
val_predictions = tree.predict(X_image_val)

# Accuracy = correct predictions / total validation samples
val_accuracy = np.mean(val_predictions == y_image_val)

print("Validation accuracy:", val_accuracy)
print("Validation accuracy (%):", val_accuracy * 100)


# Test Decision Tree on sentiment classification

# Start with 1,000 sentiment training samples
X_sent_small = X_sent_train[:2000]
y_sent_small = y_sent_train[:2000]

# Create a Decision Tree with maximum depth 5
sent_tree = DecisionTree(max_depth=5)

# Train the model
sent_tree.fit(X_sent_small, y_sent_small)

# Calculate training accuracy
sent_train_predictions = sent_tree.predict(X_sent_small)
sent_train_accuracy = np.mean(
    sent_train_predictions == y_sent_small
)

# Calculate validation accuracy on unseen samples
sent_val_predictions = sent_tree.predict(X_sent_val)
sent_val_accuracy = np.mean(
    sent_val_predictions == y_sent_val
)

print("\nSentiment training accuracy:", sent_train_accuracy)
print("Sentiment validation accuracy:", sent_val_accuracy)


