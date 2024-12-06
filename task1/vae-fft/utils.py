import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset

from Attention import multiHeadAttention
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(
        self, in_channels, patch_size, emb_size, img_size
    ):  # in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224
        super().__init__()
        self.patch_size = patch_size
        self.nPatches = (img_size * img_size) // ((patch_size) ** 2)
        self.projection = nn.Sequential(
            Rearrange(
                "b c (h p1)(w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_size * patch_size * in_channels, emb_size),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1,emb_size))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.projection(x)
        cls_tokens = repeat(
            self.cls_token, "() n e -> b n e", b=b
        )  # repeat the cls tokens for all patch set in
        x = torch.cat([cls_tokens, x], dim=1)
        # x+=self.positions
        return x


class residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        identity = x
        res = self.fn(x)
        out = res + identity
        return out


class DeepBlock(nn.Sequential):
    def __init__(self, emb_size: int = 256, drop_out: float = 0.0):  # 64
        super().__init__(
            residual(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    multiHeadAttention(emb_size, 2, drop_out),
                    nn.LayerNorm(emb_size),
                )
            )
        )


class Classification(nn.Sequential):
    def __init__(self, emb_size: int = 256, n_classes: int = 2):
        super().__init__(
            # Reduce('b n e -> b e', reduction='mean'),
            nn.Dropout(0.01),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )


def get_mag(f):
    dm_frequency_domain = np.fft.fftshift(f)
    dm_reduced_domain = dm_frequency_domain.copy()
    # Set a threshold for magnitude to retain only the most significant coefficients
    threshold = 0.001 * np.max(np.abs(dm_reduced_domain))
    dm_reduced_domain[np.abs(dm_reduced_domain) < threshold] = 0
    transformed_image = np.log(1 + np.abs(dm_reduced_domain))
    img_filtered = torch.tensor(transformed_image)
    return img_filtered


def readImage(imagePath):
    # Load image
    img = cv2.imread(imagePath)
    # img = cv2.resize(img, (256, 256))

    f1 = get_mag(np.fft.fft2(img[:, :, 0]))
    f2 = get_mag(np.fft.fft2(img[:, :, 1]))
    f3 = get_mag(np.fft.fft2(img[:, :, 2]))

    result = torch.stack([f1, f2, f3], 0)
    return result


class buildDataset(Dataset):
    def __init__(self, rootFolder):
        self.rootFolder = rootFolder
        self.images = []
        for f in os.listdir(self.rootFolder):
            if f == "FAKE":
                f = os.path.join(self.rootFolder, f)
                ind = 0
                for im in tqdm(os.listdir(f)):
                    ind += 1
                    if ind > 40000:
                        break
                    im = os.path.join(f, im)
                    img = readImage(im)
                    self.images.append([img, 0])

            else:
                f = os.path.join(self.rootFolder, f)
                ind = 0
                for im in tqdm(os.listdir(f)):
                    ind += 1
                    if ind > 40000:
                        break
                    im = os.path.join(f, im)
                    img = readImage(im)
                    self.images.append([img, 1])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        input1 = self.images[index][0]
        label = self.images[index][1]
        return (input1, label)
