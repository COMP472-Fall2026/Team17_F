import torch
from torch import nn

def train_torch_model(model, train_loader, val_loader, epochs=50, lr=0.001, weight_decay=0.0):

    loss_fn = nn.CrossEntropyLoss()                     # Define the loss function
    optimizer = torch.optim.Adam(             # Define the optimizer
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )                                                   


    # Store performance metrics for every epoch
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }


    # For each epoch:
        #Switch to training mode

        #For each training batch:
            #Make predictions
            #Calculate loss
            #Backpropagate
            #Update weights
        
        #Calculate average training loss

        #Switch to evaluation mode

        #For each validation batch:
            #Make predictions
            #Count correct predictions
        
        #Calculate validation accuracy

    best_val_accuracy = 0.0
    best_model_state = None 

    # Repeat training for the specified number of epochs (training loop)
    for epoch in range(1, epochs + 1):
        model.train()           # Switch the Neural Network to training mode
        total_loss = 0.0        # Track the total loss during this epoch
        train_correct = 0       # Track the correct training
        train_total = 0         # Track the Total training

        
        for xb, yb in train_loader:     # Process each training batch
            optimizer.zero_grad()       # Clear gradients from the previous batch
            predictions = model(xb)     # Make predictions 
            loss = loss_fn(predictions, yb) #Calculate loss
            loss.backward()        # Calculate gradients
            optimizer.step()   # Update the model's weights
            total_loss += loss.item() * len(xb)  #The batch loss


            # Calculate training accuracy
            predicted = predictions.argmax(dim=1)
            train_correct += (predicted == yb).sum().item()
            train_total += len(yb)



        # Calculate average training loss for this epoch
        avg_train_loss = total_loss / len(train_loader.dataset)
        # Calculate average traning accuracy 
        train_accuracy = train_correct / train_total
        print(f"Epoch {epoch}/{epochs} - Training Loss: {avg_train_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")


        model.eval() # Switch to evaluation mode

        correct = 0
        total = 0

    
        with torch.no_grad(): # Evaluate without calculating gradients
            
            for xb, yb in val_loader:
                outputs = model(xb) # Make predictions
                predicted = outputs.argmax(dim=1) # Get the predicted class

                # Count correct predictions
                correct += (predicted == yb).sum().item()
                total += len(yb)

        # Calculate validation accuracy
        val_accuracy = correct / total
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        #This checks whether the current epoch achieved 
        # a higher validation accuracy and saves a copy of its weights.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            best_model_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }


        # Save this epoch's results
        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

    #restores the weights from the best epoch after training finishes.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history





