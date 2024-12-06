import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = timm.create_model("tiny_vit_21m_224.dist_in22k", pretrained=True)
model.head.fc = nn.Linear(model.head.fc.in_features, 1)
model.load_state_dict(torch.load("model2.pth"))
model = model.to(device)

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
test_transforms = timm.data.create_transform(**data_config, is_training=False)


# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = dict(self.model.named_modules())[self.target_layer_name]

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backpropagate
        self.model.zero_grad()
        output[:, class_idx].backward(retain_graph=True)

        # Compute Grad-CAM
        gradients = self.gradients.cpu().detach()
        activations = self.activations.cpu().detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = torch.sum(weights * activations, dim=1).squeeze()
        grad_cam = F.relu(grad_cam)

        # Normalize the Grad-CAM output
        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()
        return grad_cam.numpy()


# Helper function to overlay Grad-CAM on an image
def gradcam(image, grad_cam, alpha=0.5, cmap="jet"):
    grad_cam_resized = (
        F.interpolate(
            torch.tensor(grad_cam).unsqueeze(0).unsqueeze(0),
            size=image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )

    heatmap = plt.get_cmap(cmap)(grad_cam_resized)[..., :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    return heatmap


# Grad-CAM for high-level feature map
target_layer = "stages.3.blocks.1.local_conv"
grad_cam = GradCAM(model, target_layer)


def merge_original_and_gradcam(
    original_img_path, gradcam_output, alpha=0.5, cmap="jet"
):
    original_img = cv2.imread(original_img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    gradcam_resized = cv2.resize(
        gradcam_output, (original_img.shape[1], original_img.shape[0])
    )

    gradcam_resized = (gradcam_resized * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(gradcam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
    return overlay


for img in tqdm(os.listdir("task2_images")):
    img_path = os.path.join("task2_images", img)
    image = Image.open(img_path).convert("RGB")
    image = test_transforms(image).unsqueeze(0).to(device)

    grad_cam_output = grad_cam.generate(image)
    image = plt.imread(img_path)
    grad_cam_output = gradcam(image, grad_cam_output)
    overlay = merge_original_and_gradcam(img_path, grad_cam_output)

    plt.imsave(f"gradcam_images/{img}", overlay)
