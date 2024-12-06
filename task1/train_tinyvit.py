import warnings
import sklearn.exceptions

import glob

from PIL import Image

import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("tiny_vit_21m_224.dist_in22k", pretrained=True)
model.head.fc = nn.Linear(model.head.fc.in_features, 1)
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)
train_transforms = timm.data.create_transform(**data_config, is_training=True)
test_transforms = timm.data.create_transform(**data_config, is_training=False)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = glob.glob(f"{root_dir}/REAL/*.jpg") + glob.glob(
            f"{root_dir}/FAKE/*.jpg"
        )
        self.labels = [0] * len(glob.glob(f"{root_dir}/REAL/*.jpg")) + [1] * len(
            glob.glob(f"{root_dir}/FAKE/*.jpg")
        )
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.float32)


train_ds = CustomDataset(root_dir="../train", transform=train_transforms)
test_ds = CustomDataset(root_dir="../test", transform=test_transforms)

train_loader = DataLoader(
    train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

epochs = 10
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.reshape(-1, 1)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")

    scheduler.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()}")

torch.save(model.state_dict(), "model.pth")

preds = []
actual = []
model.eval()

for i, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    outputs = torch.sigmoid(outputs)
    preds.extend((outputs > 0.5).cpu().numpy())
    actual.extend(labels.cpu().numpy())


print(classification_report(actual, preds))
