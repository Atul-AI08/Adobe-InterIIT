import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from Encoder import *
from EarlyStopping import EarlyStopping
from model import Model
from sklearn.metrics import confusion_matrix, classification_report


def trainModel(model, criterion, optimizer, epochs, trainDs, valDs, path, patience=5):
    best_model_wts = copy.deepcopy(model.state_dict())
    batch_train_loss = []
    batch_train_acc = []
    batch_val_acc = []
    batch_val_loss = []
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, delta=0.01, path=path
    )

    max_acc = 0
    for epoch in tqdm(range(epochs)):
        # Training phase
        model.train()
        current_corrects = 0.0
        train_loss = []
        for inputs1, labels in trainDs:
            input1 = inputs1.to(device).float()
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(input1)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            current_corrects += torch.sum(preds == labels.data)

        train_acc = current_corrects.double() / len(trainDs.dataset)
        batch_train_loss.append(np.mean(train_loss))
        batch_train_acc.append(train_acc)

        # Validation phase
        model.eval()
        current_corrects = 0.0
        val_loss = []
        with torch.no_grad():
            for inputs1, labels in valDs:
                input1 = inputs1.to(device).float()
                labels = labels.to(device)
                outputs = model(input1)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())
                current_corrects += torch.sum(preds == labels.data)

        val_acc = current_corrects.double() / len(valDs.dataset)
        batch_val_loss.append(np.mean(val_loss))
        batch_val_acc.append(val_acc)

        if max_acc < val_acc:
            best_model_wts = copy.deepcopy(model.state_dict())

        early_stopping(-1 * val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print(
            f"Epoch {epoch}: Train Loss {batch_train_loss[-1]:.4f}, Acc: {batch_train_acc[-1]:.4f}"
        )
        print(
            f"Epoch {epoch}: Val Loss {batch_val_loss[-1]:.4f}, Acc: {batch_val_acc[-1]:.4f}"
        )

    return (
        model,
        batch_train_loss,
        batch_train_acc,
        batch_val_loss,
        batch_val_acc,
        best_model_wts,
    )


# Paths
train_path = "train"
val_path = "val"
save_path = "best_model.pth"
test_dataset = "test"
test_path = "test"

# Dataset and DataLoader
train_dataset = buildDataset(train_path)
val_dataset = buildDataset(val_path)
test_dataset = buildDataset(test_path)

trainDs = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=False)
valDs = DataLoader(val_dataset, batch_size=16, shuffle=True, pin_memory=False)
testDs = DataLoader(test_dataset, batch_size=16, shuffle=True, pin_memory=False)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, Criterion, Optimizer
model = Model(16, 0, 2, 3, 8, 8, 32, device=device)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Train the Model
model, train_loss, train_acc, val_loss, val_acc, best_wts = trainModel(
    model,
    criterion,
    optimizer,
    epochs=50,
    trainDs=trainDs,
    valDs=valDs,
    path=save_path,
    patience=5,
)


# Set model to evaluation mode
model.eval()
current_corrects = 0.0
val_loss = []
all_preds = []
all_labels = []

for batchNum, (inputs1, labels) in enumerate(testDs):
    input1 = inputs1.to(device).float()
    labels = labels.to(device)

    outputs = model(input1)
    _, preds = torch.max(outputs, 1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

    loss = criterion(outputs, labels)
    val_loss.append(loss.item())

    current_corrects += torch.sum(preds == labels.data)

# Calculate overall accuracy
val_acc = current_corrects.double() / len(testDs.dataset)
print("Test Accuracy:", val_acc.item())

# Calculate confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1])
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(
    all_labels, all_preds, labels=[0, 1], target_names=["Class 0", "Class 1"]
)
print("Classification Report:")
print(class_report)
