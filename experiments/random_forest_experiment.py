from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.random_forest import RandomForest


SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Selected settings for each dataset
SETTINGS = {
    "image": {
        "n_trees": 100,
        "max_depth": 16,
        "max_features": "sqrt",
    },
    "sentiment": {
        "n_trees": 50,
        "max_depth": 8,
        "max_features": 50,
    },
}


def run_experiment(dataset, settings):
    # Load the training file for this dataset
    data_path = PROJECT_ROOT / "data" / f"{dataset}_train.csv"
    print(f"\nLoading provided file: {data_path.name}", flush=True)
    data = pd.read_csv(data_path)

    # Separate the input features from the correct labels
    X = data.drop(columns=["id", "label"]).to_numpy()
    y = data["label"].to_numpy()

    # Use the split from the selected experiment
    # Images: 85/15. Sentiment: 80/20
    validation_size = 0.15 if dataset == "image" else 0.20

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=SEED,
        stratify=y,
    )

    print("Training examples:", len(X_train))
    print("Validation examples:", len(X_val))
    print("Features per example:", X.shape[1])

    # Build a separate forest using this dataset's settings
    model = RandomForest(
        random_state=SEED,
        **settings,
    )
    print(
        f"Training {model.n_trees} trees "
        f"(depth {model.max_depth}, "
        f"max_features={model.max_features})...",
        flush=True,
    )

    # Train the forest and measure the training time
    start = perf_counter()
    model.fit(X_train, y_train)
    training_seconds = perf_counter() - start

    # Check predictions on training and validation examples
    print("Training complete. Measuring accuracy...", flush=True)
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    # Calculate the proportion of correct predictions
    train_accuracy = np.mean(train_predictions == y_train)
    val_accuracy = np.mean(val_predictions == y_val)

    print(f"Training accuracy: {train_accuracy:.2%}")
    print(f"Validation accuracy: {val_accuracy:.2%}")
    print(f"Training time: {training_seconds:.2f} seconds")


def main():
    # Check both files before starting the training
    for dataset in SETTINGS:
        path = PROJECT_ROOT / "data" / f"{dataset}_train.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Training CSV not found: {path}")

    # Run images first, then sentiment
    for dataset, settings in SETTINGS.items():
        run_experiment(dataset, settings)


if __name__ == "__main__":
    main()
