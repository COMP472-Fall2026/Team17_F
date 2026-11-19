import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np

from models.neural_network import NeuralNetwork
from torch.utils.data import TensorDataset, DataLoader

from models.neural_network_training import train_torch_model


# Load the image training dataset
image_data = pd.read_csv("data/image_train.csv")

# Check the dataset dimensions
print("Dataset shape:", image_data.shape)

# Separate the input features and target labels
X = image_data.drop(columns=["id", "label"])
y = image_data["label"]
print("Minimum:", X.min().min())
print("Maximum:", X.max().max())
print("Mean:", X.mean().mean())
print("Standard deviation:", X.std().mean())

feature_norms = np.linalg.norm(X.to_numpy(), axis=1)

print("Minimum L2 norm:", feature_norms.min())
print("Maximum L2 norm:", feature_norms.max())
print("Mean L2 norm:", feature_norms.mean())

# Check the dimensions
print("Features shape:", X.shape)
print("Labels shape:", y.shape)


# Split the image dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert input features into PyTorch tensors
X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)

# Convert labels into PyTorch tensors
y_train = torch.tensor(y_train.to_numpy(), dtype=torch.long)
y_val = torch.tensor(y_val.to_numpy(), dtype=torch.long)

# Verify tensor dimensions
print("Training tensor:", X_train.shape)
print("Validation tensor:", X_val.shape)
print("Training labels tensor:", y_train.shape)
print("Validation labels tensor:", y_val.shape)


# Combine features and labels into datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


train_generator = torch.Generator()
train_generator.manual_seed(42)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    generator=train_generator
)


val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

# Check the number of batches
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


#controlling three sources of randomness: 
    # the dataset split, 
    # the model initialization, 
    # and the batch order.


# Experiment 1: Learning Rate = 0.001
def experiment_1():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_1 = NeuralNetwork(
        input_size=500,
        num_classes=50
    )

    history_1 = train_torch_model(
        model_1,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001
    )

    best_accuracy = max(history_1["val_accuracy"])
    best_epoch = history_1["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 1: Learning Rate = 0.001")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)
    print("Final Training Accuracy:", history_1["train_accuracy"][-1])


# Experiment 2: Learning Rate = 0.0001
def experiment_2():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_2 = NeuralNetwork(
        input_size=500,
        num_classes=50
    )

    history_2 = train_torch_model(
        model_2,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.0001
    )

    best_accuracy = max(history_2["val_accuracy"])
    best_epoch = history_2["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 2: Learning Rate = 0.0001")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)
    print("Final Training Accuracy:", history_2["train_accuracy"][-1])




# Experiment 3: Dropout = 0.3
def experiment_3():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_3 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.3
    )

    history_3 = train_torch_model(
        model_3,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001
    )

    best_accuracy = max(history_3["val_accuracy"])
    best_epoch = history_3["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 3: Dropout 0.3")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)

    torch.save(
        model_3.state_dict(),
        "best_neural_network.pth"
    )


# Experiment 4: Dropout = 0.5
def experiment_4():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_4 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.5
    )

    history_4 = train_torch_model(
        model_4,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001
    )

    best_accuracy = max(history_4["val_accuracy"])
    best_epoch = history_4["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 4: Dropout 0.5")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)


# Experiment 5: Dropout 0.3 + Weight Decay
def experiment_5():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_5 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.3
    )

    history_5 = train_torch_model(
        model_5,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=0.0001
    )

    best_accuracy = max(history_5["val_accuracy"])
    best_epoch = history_5["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 5: Dropout 0.3 + Weight Decay 0.0001")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)


# Experiment 6: Smaller Neural Network
def experiment_6():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_6 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.3,
        hidden_size=128
    )

    history_6 = train_torch_model(
        model_6,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=0.0
    )

    best_accuracy = max(history_6["val_accuracy"])
    best_epoch = history_6["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 8: Smaller Neural Network")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)


# Experiment 7: Dropout 0.4
def experiment_7():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_7 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.4,
        hidden_size=256
    )

    history_7 = train_torch_model(
        model_7,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=0.0
    )

    best_accuracy = max(history_7["val_accuracy"])
    best_epoch = history_7["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 9: Dropout 0.4")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)

    torch.save(
        model_7.state_dict(),
        "best_neural_network_dropout_04.pth"
    )


# Experiment 8: Dropout 0.2
def experiment_8():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_8 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.2,
        hidden_size=256
    )

    history_8 = train_torch_model(
        model_8,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=0.0
    )

    best_accuracy = max(history_8["val_accuracy"])
    best_epoch = history_8["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 10: Dropout 0.2")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)




# Experiment 9: Dropout 0.4 + Learning Rate 0.0005
def experiment_9():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_9 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.4,
        hidden_size=256
    )

    history_9 = train_torch_model(
        model_9,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.0005,
        weight_decay=0.0
    )

    best_accuracy = max(history_9["val_accuracy"])
    best_epoch = history_9["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 9: Dropout 0.4 + LR 0.0005")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)


# Experiment 10: Dropout 0.4 + Learning Rate 0.002
def experiment_10():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_10 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.4,
        hidden_size=256
    )

    history_10 = train_torch_model(
        model_10,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.002,
        weight_decay=0.0
    )

    best_accuracy = max(history_10["val_accuracy"])
    best_epoch = history_10["val_accuracy"].index(best_accuracy) + 1

    print("\nExperiment 10: Dropout 0.4 + LR 0.002")
    print("Best Validation Accuracy:", best_accuracy)
    print("Best Epoch:", best_epoch)



# Experiment 14: Batch Normalization
def experiment_11():
    torch.manual_seed(42)
    train_generator.manual_seed(42)

    model_11 = NeuralNetwork(
        input_size=500,
        num_classes=50,
        dropout=0.4,
        hidden_size=256,
        use_batch_norm=True
    )

    history_11 = train_torch_model(
        model_11,
        train_loader,
        val_loader,
        epochs=50,
        lr=0.001,
        weight_decay=0.0
    )

    best_accuracy_14 = max(history_11["val_accuracy"])
    best_epoch_14 = history_11["val_accuracy"].index(best_accuracy_14) + 1

    print("\nExperiment 14: Batch Normalization")
    print("Best Validation Accuracy:", best_accuracy_14)
    print("Best Epoch:", best_epoch_14)
    print("Final Training Accuracy:", history_11["train_accuracy"][-1])

    torch.save(
        model_11.state_dict(),
        "best_neural_network_batch_norm.pth"
    )


if __name__ == "__main__":
    experiment = 11

    if experiment == 1:
        experiment_1()

    elif experiment == 2:
        experiment_2()

    elif experiment == 3:
        experiment_3()

    elif experiment == 4:
        experiment_4()

    elif experiment == 5:
        experiment_5()

    elif experiment == 6:
        experiment_6()

    elif experiment == 7:
        experiment_7()

    elif experiment == 8:
        experiment_8()
    
    elif experiment == 9:
        experiment_9()

    elif experiment == 10:
        experiment_10()

    elif experiment == 11:
        experiment_11()
    else:
        print("Invalid experiment number.")
