import warnings
import sklearn.exceptions

import glob

from PIL import Image
import pandas as pd

import torch
import timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = timm.create_model("tiny_vit_21m_224.dist_in22k", pretrained=True)
model.head.fc = nn.Linear(model.head.fc.in_features, 1)
model.load_state_dict(torch.load("model.pth"))
model = model.to(device)

data_config = timm.data.resolve_model_data_config(model)
test_transforms = timm.data.create_transform(**data_config, is_training=False)


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = glob.glob(f"{root_dir}/*.png")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, self.images[idx]


test_ds = CustomDataset(root_dir="final_test", transform=test_transforms)

test_loader = DataLoader(
    test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)

df = pd.DataFrame(columns=["image", "pred"])
model.eval()

for i, (images, img_path) in enumerate(test_loader):
    images = images.to(device)
    outputs = model(images)
    outputs = torch.sigmoid(outputs)
    preds = (outputs > 0.5).float()
    for j in range(len(img_path)):
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {
                        "image": [img_path[j]],
                        "pred": ["Fake" if preds[j].item() else "Real"],
                    }
                ),
            ]
        )

df.to_csv("submission.csv", index=False)
