import os
from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define image size and normalization parameters
IMAGE_SIZE = 224
mean, std = [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]

# Define transformations for training and validation datasets
composed_train = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.1),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False),
    ]
)

composed_test = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# Define directories for your datasets
train_dir = "/workspace/SD_CIFAKE/"
test_dir = "/workspace/MYK_ATTACKED/"

# Load datasets using ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir, transform=composed_train)
validation_dataset = datasets.ImageFolder(root=test_dir, transform=composed_test)

# Create data loaders
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=100, shuffle=True
)
validation_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset, batch_size=100, shuffle=False
)


# Define the ResNet-34 model
def resnet_34():
    resnet = torchvision.models.resnet34(pretrained=True)
    resnet.fc = torch.nn.Linear(
        resnet.fc.in_features, len(train_dataset.classes)
    )  # Adjust for the number of classes
    torch.nn.init.xavier_uniform_(resnet.fc.weight)
    return resnet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet_34()
model.to(device)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.2)
scheduler = ReduceLROnPlateau(optimizer, "min")


# Function to train the model
def train_model(model, n_epochs, train_loader, validation_loader, optimizer, scheduler):
    train_cost_list = []
    val_cost_list = []
    accuracy_list = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_cost_list.append(running_loss / len(train_loader))

        # Validate the model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(validation_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_cost_list.append(val_loss / len(validation_loader))
        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)

        print(
            f"Validation Loss: {val_loss / len(validation_loader):.4f}, Accuracy: {accuracy:.2f}%"
        )

        # Step the scheduler
        scheduler.step(val_loss / len(validation_loader))

    return accuracy_list, train_cost_list, val_cost_list


# Train the model
n_epochs = 5

accuracy_list, train_cost_list, val_cost_list = train_model(
    model=model,
    n_epochs=n_epochs,
    train_loader=train_loader,
    validation_loader=validation_loader,
    optimizer=optimizer,
    scheduler=scheduler,
)

# Save the trained model
save_dir = "/workspace/results/resnet_finetuned"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "resnet34_finetuned.pth"))
print(f"Model saved to {save_dir}")


def evaluate_model(model, data_loader, criterion):
    model.to(device)
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Calculate predictions
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(data_loader)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# Evaluate the model on the test set
test_loss, test_accuracy = evaluate_model(model, validation_loader, criterion)
