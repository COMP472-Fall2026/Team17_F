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
# Standardize the features
#scaler = StandardScaler()

#X_train = scaler.fit_transform(X_train)
#X_val = scaler.transform(X_val)
    #StandardScaler() creates the scaler.
    #fit_transform(X_train) calculates the mean and standard deviation of each feature 
        # using only the training data, then standardizes it.
    #transform(X_val) standardizes the validation data using those same training statistics.
    #This avoids data leakage, because the validation data doesn't influence the scaling parameters.

# Convert standardized features into PyTorch tensors
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

# Experiment 1: Learning rate = 0.001
#torch.manual_seed(42)
#model_1 = NeuralNetwork(input_size=500, num_classes=50)

#train_generator.manual_seed(42)

#history_1 = train_torch_model(
    #model_1,
    #train_loader,
    #val_loader,
    #epochs=50,
    #lr=0.001
#)

# Experiment 2: Learning rate = 0.0001
#torch.manual_seed(42)
#model_2 = NeuralNetwork(input_size=500, num_classes=50)

#train_generator.manual_seed(42)

#history_2 = train_torch_model(
    #model_2,
    #train_loader,
    #val_loader,
    #epochs=50,
    #lr=0.0001
#)

# Summarize the two learning-rate experiments
#for lr, history in [(0.001, history_1), (0.0001, history_2)]:

    #best_accuracy = max(history["val_accuracy"])
    #best_epoch = history["val_accuracy"].index(best_accuracy) + 1

    #print(f"\nLearning Rate: {lr}")
    #print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    #print(f"Best Epoch: {best_epoch}")
    #print(f"Final Training Accuracy: {history['train_accuracy'][-1]:.4f}")
    #print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")


# Experiment 3: Neural Network with Dropout
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_3 = NeuralNetwork(
    #input_size=500,
    #num_classes=50,
    #dropout=0.3
#)

#history_3 = train_torch_model(
    #model_3,
    #train_loader,
    #val_loader,
    #epochs=50,
    #lr=0.001
#)

#torch.save(model_3.state_dict(), "best_neural_network.pth")

#best_accuracy = max(history_3["val_accuracy"])
#best_epoch = history_3["val_accuracy"].index(best_accuracy) + 1

#print("Dropout 0.3 - Best Validation Accuracy:", best_accuracy)
#print("Best Epoch:", best_epoch)


# Experiment 4: Dropout 0.5
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_4 = NeuralNetwork(
    #input_size=500,
    #num_classes=50,
    #dropout=0.5
#)

#history_4 = train_torch_model(
    #model_4,
    #train_loader,
    #val_loader,
    #epochs=50,
    #lr=0.001
#)

#best_accuracy_4 = max(history_4["val_accuracy"])
#best_epoch_4 = history_4["val_accuracy"].index(best_accuracy_4) + 1

#print("Dropout 0.5 - Best Validation Accuracy:", best_accuracy_4)
#print("Best Epoch:", best_epoch_4)

# Experiment 5: Dropout 0.3 + Weight Decay
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_5 = NeuralNetwork(
    #input_size=500,
    #num_classes=50,
    #dropout=0.3
#)

#history_5 = train_torch_model(
    #model_5,
    #train_loader,
    #val_loader,
    #epochs=50,
    #lr=0.001,
    #weight_decay=0.0001
#)

#best_accuracy_5 = max(history_5["val_accuracy"])
#best_epoch_5 = history_5["val_accuracy"].index(best_accuracy_5) + 1

#print("Dropout 0.3 + Weight Decay 0.0001")
#print("Best Validation Accuracy:", best_accuracy_5)
#print("Best Epoch:", best_epoch_5)

# Experiment 6: Dropout 0.3 + Standardized Features
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_6 = NeuralNetwork(
    #input_size=500,
    #num_classes=50,
    #dropout=0.3
#)

#history_6 = train_torch_model(
    #model_6,
    #train_loader,
    #val_loader,
    #epochs=50,
    #lr=0.001,
    #weight_decay=0.0
#)

#best_accuracy_6 = max(history_6["val_accuracy"])
#best_epoch_6 = history_6["val_accuracy"].index(best_accuracy_6) + 1

#print("Experiment 6: Standardized Features")
#print("Best Validation Accuracy:", best_accuracy_6)
#print("Best Epoch:", best_epoch_6)

# Experiment 7: Standardized Features + Lower Learning Rate
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_7 = NeuralNetwork(
    #input_size=500,
    #num_classes=50,
    #dropout=0.3
#)

#history_7 = train_torch_model(
    #model_7,
    #train_loader,
    #val_loader,
    #epochs=50,
    #lr=0.0001,
    #weight_decay=0.0
#)

#best_accuracy_7 = max(history_7["val_accuracy"])
#best_epoch_7 = history_7["val_accuracy"].index(best_accuracy_7) + 1

#print("Experiment 7: Standardized + LR 0.0001")
#print("Best Validation Accuracy:", best_accuracy_7)
#print("Best Epoch:", best_epoch_7)

# Experiment 8: Smaller Neural Network
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_8 = NeuralNetwork(
#    input_size=500,
#    num_classes=50,
#    dropout=0.3,
#    hidden_size=128
#)

#history_8 = train_torch_model(
#    model_8,
#    train_loader,
#    val_loader,
#    epochs=50,
#    lr=0.001,
#    weight_decay=0.0
#)

#best_accuracy_8 = max(history_8["val_accuracy"])
#best_epoch_8 = history_8["val_accuracy"].index(best_accuracy_8) + 1

#print("Experiment 8: Smaller Neural Network")
#print("Best Validation Accuracy:", best_accuracy_8)
#print("Best Epoch:", best_epoch_8)

# Experiment 9: Dropout 0.4
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_9 = NeuralNetwork(
#    input_size=500,
#    num_classes=50,
#    dropout=0.4,
#    hidden_size=256
#)

#history_9 = train_torch_model(
#    model_9,
#    train_loader,
#    val_loader,
#    epochs=50,
#    lr=0.001,
#    weight_decay=0.0
#)

#best_accuracy_9 = max(history_9["val_accuracy"])
#best_epoch_9 = history_9["val_accuracy"].index(best_accuracy_9) + 1

#print("Experiment 9: Dropout 0.4")
#print("Best Validation Accuracy:", best_accuracy_9)
#print("Best Epoch:", best_epoch_9)

#torch.save(
#    model_9.state_dict(),
#    "best_neural_network_dropout_04.pth"
#)

# Experiment 10: Dropout 0.2
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_10 = NeuralNetwork(
#    input_size=500,
#    num_classes=50,
#    dropout=0.2,
#    hidden_size=256
#)

#history_10 = train_torch_model(
#    model_10,
#    train_loader,
#    val_loader,
#    epochs=50,
#    lr=0.001,
#    weight_decay=0.0
#)

#best_accuracy_10 = max(history_10["val_accuracy"])
#best_epoch_10 = history_10["val_accuracy"].index(best_accuracy_10) + 1

#print("Experiment 10: Dropout 0.2")
#print("Best Validation Accuracy:", best_accuracy_10)
#print("Best Epoch:", best_epoch_10)

# Experiment 11: Dropout 0.4, Learning Rate 0.0005
# Experiment 12: Dropout 0.4, Learning Rate 0.002
#torch.manual_seed(42)
#train_generator.manual_seed(42)

#model_12 = NeuralNetwork(
#    input_size=500,
#    num_classes=50,
#    dropout=0.4,
#    hidden_size=256
#)

#history_12 = train_torch_model(
#    model_12,
#    train_loader,
#    val_loader,
#    epochs=50,
#    lr=0.002,
#    weight_decay=0.0
#)

#best_accuracy_12 = max(history_12["val_accuracy"])
#best_epoch_12 = history_12["val_accuracy"].index(best_accuracy_12) + 1

#print("Experiment 12: Dropout 0.4, LR 0.002")
#print("Best Validation Accuracy:", best_accuracy_12)
#print("Best Epoch:", best_epoch_12)

#Experiment 14: Batch Normalization
torch.manual_seed(42)
train_generator.manual_seed(42)

model_14 = NeuralNetwork(
    input_size=500,
    num_classes=50,
    dropout=0.4,
    hidden_size=256,
    use_batch_norm=True
)

history_14 = train_torch_model(
    model_14,
    train_loader,
    val_loader,
    epochs=50,
    lr=0.001,
    weight_decay=0.0
)

best_accuracy_14 = max(history_14["val_accuracy"])
best_epoch_14 = history_14["val_accuracy"].index(best_accuracy_14) + 1

print("Experiment 14: Batch Normalization")
print("Best Validation Accuracy:", best_accuracy_14)
print("Best Epoch:", best_epoch_14)
print("Final Training Accuracy:", history_14["train_accuracy"][-1])

torch.save(
    model_14.state_dict(),
    "best_neural_network_batch_norm.pth"
)
# Experiment 15: Batch Normalization + Dropout 0.3 --> validation accuracy dropped, training accuracy increased
