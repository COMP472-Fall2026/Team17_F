import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.neural_network import NeuralNetwork
from models.neural_network_training import train_torch_model


# Set paths relative to the project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "neural_network_sentiment"
MODEL_PATH = OUTPUT_DIR / "best_model.pth"

# Settings used for the selected sentiment model
SEED = 42
VALIDATION_SIZE = 0.20
LEARNING_RATE = 0.0001
EPOCHS = 20
BATCH_SIZE = 64
FEATURE_COLUMNS = [f"feature_{i}" for i in range(768)]


def create_model():
    # Use the same network for training and test predictions
    return NeuralNetwork(
        input_size=768,
        num_classes=3,
        hidden_size=256,
        dropout=0.0,
        use_batch_norm=False,
    )


def load_sentiment_split():
    # Load the sentiment training dataset
    data = pd.read_csv(
        PROJECT_ROOT / "data" / "sentiment_train.csv",
        dtype={"id": str},
    )

    # Separate the input features correct labels and sample IDs
    X = data[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = data["label"].to_numpy(dtype="int64")
    ids = data["id"].to_numpy()

    # Keep 20% for validation, with the same class proportions
    return train_test_split(
        X,
        y,
        ids,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y,
    )


def evaluate_model(model, loader):
    # Switch to evaluation mode
    model.eval()
    correct = 0
    total = 0
    predictions = []

    # Evaluate without calculating gradients for training updates
    with torch.no_grad():
        for features, labels in loader:
            # Choose the class with the highest score
            predicted = model(features).argmax(dim=1)

            # Count correct predictions and store the predicted labels
            correct += (predicted == labels).sum().item()
            total += len(labels)
            predictions.append(predicted.cpu())

    # Return the accuracy and combine predictions from all batches
    return correct / total, torch.cat(predictions).numpy()


def train_sentiment_model(X_train, X_val, y_train, y_val, val_ids):
    # Combine features and labels into PyTorch datasets
    train_dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    # Use this loader to check training accuracy after training finishes
    train_eval_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Keep validation predictions in the same order as their IDs
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Control the starting weights and training batch order
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)

    # Shuffle the training examples and process them in batches
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    model = create_model()

    # Create the output folder if it does not exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Learning rate: {LEARNING_RATE:g} | "
        f"Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}",
        flush=True,
    )

    # Train the model and measure the training time
    start = perf_counter()
    history = train_torch_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    training_seconds = perf_counter() - start

    # The training function restores the best epoch's weights
    # Check both accuracies using those same weights
    train_accuracy, _ = evaluate_model(model, train_eval_loader)
    val_accuracy, val_predictions = evaluate_model(model, val_loader)

    # Find the epoch with the highest validation accuracy
    best_epoch = history["val_accuracy"].index(
        max(history["val_accuracy"])
    ) + 1

    # Save the trained weights for later predictions
    torch.save(model.state_dict(), MODEL_PATH)

    # Save the results from each epoch
    history_table = pd.DataFrame(history)
    history_table.insert(0, "epoch", range(1, len(history_table) + 1))
    history_table.to_csv(
        OUTPUT_DIR / f"history_lr_{LEARNING_RATE:g}.csv",
        index=False,
    )

    # Save a summary of this training run
    summary = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "best_epoch": best_epoch,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "training_seconds": training_seconds,
    }
    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "results.csv", index=False
    )

    # Save validation predictions so we can examine the mistakes
    pd.DataFrame({
        "id": val_ids,
        "true_label": y_val,
        "predicted_label": val_predictions,
    }).to_csv(
        OUTPUT_DIR / "validation_predictions.csv", index=False
    )

    # Display the results and confusion matrix
    print(f"Best epoch: {best_epoch}")
    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation accuracy: {val_accuracy:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print("Confusion matrix (rows = actual, columns = predicted)")
    print("Order: negative, neutral, positive")
    print(confusion_matrix(y_val, val_predictions, labels=[0, 1, 2]))
    print("Results and best model saved in:", OUTPUT_DIR)


def export_sentiment_predictions():
    # Check that the trained model is available
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Missing best_model.pth. Run with --train first."
        )

    # Load the sentiment test dataset
    test = pd.read_csv(
        PROJECT_ROOT / "data" / "sentiment_test.csv",
        dtype={"id": str},
    )

    # Check the number of test examples and their IDs
    if len(test) != 5000:
        raise ValueError("Expected 5,000 sentiment test rows.")
    if test["id"].isna().any() or test["id"].duplicated().any():
        raise ValueError("Missing or duplicated test IDs.")

    # Convert the input features into a PyTorch tensor
    X = torch.from_numpy(
        test[FEATURE_COLUMNS].to_numpy(dtype="float32")
    )
    if not torch.isfinite(X).all():
        raise ValueError("Test features contain invalid values.")

    # Recreate the network and load the saved weights
    model = create_model()
    weights = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(weights)
    model.eval()

    # Predict the test labels in batches
    predictions = []
    with torch.no_grad():
        for batch in X.split(BATCH_SIZE):
            scores = model(batch)
            if not torch.isfinite(scores).all():
                raise ValueError("The model produced invalid scores.")
            predictions.append(scores.argmax(dim=1))

    # Match each predicted label with its test ID
    result = pd.DataFrame({
        "id": test["id"],
        "label": torch.cat(predictions).cpu().numpy(),
    })

    # Save the predictions without an extra index column
    path = OUTPUT_DIR / "sentiment_predictions.csv"
    result.to_csv(path, index=False)

    print(f"Saved {len(result)} predictions to: {path}")
    print(result.head())


def main():
    # Read the --train and --export options from the terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the sentiment model using the selected settings.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export sentiment test predictions.",
    )
    args = parser.parse_args()

    # Export predictions using the saved model, then stop
    if args.export:
        export_sentiment_predictions()
        return

    # Load the training and validation sets
    X_train, X_val, y_train, y_val, _train_ids, val_ids = (
        load_sentiment_split()
    )
    print("Training inputs:", X_train.shape)
    print("Validation inputs:", X_val.shape)

    # Start training when --train is provided
    if args.train:
        train_sentiment_model(
            X_train, X_val, y_train, y_val, val_ids
        )
    else:
        print("Data check complete. Add --train to train the neural network.")


if __name__ == "__main__":
    main()
