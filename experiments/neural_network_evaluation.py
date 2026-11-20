import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from models.neural_network import NeuralNetwork

# Create the same architecture as our best model (Experiment 14)
model = NeuralNetwork(
    input_size=500,
    num_classes=50,
    dropout=0.4,
    hidden_size=256,
    use_batch_norm=True
)

# Load the saved model weights
model.load_state_dict(
    torch.load(
        "best_neural_network_batch_norm.pth",
        map_location="cpu",
        weights_only=True
    )
)


# Switch to evaluation mode
model.eval()

print("Best Neural Network loaded successfully!")

# Load the original image dataset
image_data = pd.read_csv("data/image_train.csv")

# Separate features and labels
X = image_data.drop(columns=["id", "label"])
y = image_data["label"]

# Recreate the same training-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert validation features and labels into tensors
X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)

# Verify the dimensions
print("Validation features:", X_val.shape)
print("Validation labels:", y_val.shape)

# Evaluate the saved Neural Network
with torch.no_grad():

    # Get the model's output scores
    outputs = model(X_val)

    # Select the class with the highest score
    predictions = torch.argmax(outputs, dim=1)

# Calculate validation accuracy
correct = (predictions == y_val).sum().item()
total = y_val.size(0)

accuracy = correct / total

print("Correct predictions:", correct)
print("Total predictions:", total)
print(f"Validation Accuracy: {accuracy:.4f}")

