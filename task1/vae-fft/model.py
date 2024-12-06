import torch
import torch.nn.functional as F

from utils import *
from Encoder import *


class Model(nn.Module):
    def __init__(
        self,
        emb_size,
        drop_out,
        n_classes,
        in_channels,
        patch_size1,
        patch_size2,
        image_size,
        device,
    ):
        super().__init__()
        self.encoder = VAE_Encoder()
        self.PatchEmbedding1 = PatchEmbedding(32, patch_size1, emb_size, 64)
        self.PatchEmbedding2 = PatchEmbedding(32, patch_size2, emb_size, 64)
        self.DeepBlock = DeepBlock(emb_size=16)  # Transformer()
        self.device = device
        self.Classification = Classification(emb_size=16, n_classes=2)

    def forward(self, x):
        x = self.encoder(x)
        #     print(x.shape)
        # x = x.view(16,32,56)
        patch1 = self.PatchEmbedding1(x)
        patch2 = self.PatchEmbedding2(x)
        #     print(patch1.shape)
        #     print(patch2.shape)
        #      Resize tensor2 along the second dimension to match the size of tensor1
        desired_size = patch1.shape[1]  # patchEmbeddings1
        indices = torch.linspace(
            0, patch2.shape[1] - 1, desired_size, device=self.device
        ).long()
        patch2_resized = torch.index_select(patch2, 1, indices)  # .to(device)
        #     print(patch2_resized.shape)

        # Concatenate the tensors along the second dimension (dim=1)
        patchEmbeddings = torch.cat((patch2_resized, patch1), dim=1)  # .to(device)
        #     print(patchEmbeddings.shape)

        DeepBlockOp = self.DeepBlock(patchEmbeddings)
        #     print(DeepBlockOp.shape)
        classificationOutput = self.Classification(DeepBlockOp)
        #     print(classificationOutput.shape)
        output = F.log_softmax(classificationOutput, dim=1)
        return output
