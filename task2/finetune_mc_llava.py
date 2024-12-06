import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerator import Accelerator

from modelling_llava import LlavaForCausalLM

# accelerator = Accelerator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = LlavaForCausalLM.from_pretrained(
    "visheratin/MC-LLaVA-3b", torch_dtype=torch.float16, trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "visheratin/MC-LLaVA-3b", trust_remote_code=True
)

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=None,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


class MyDataset(Dataset):
    def __init__(
        self,
        data,
        processor,
        max_crops=500,
        num_tokens=768,
        base_folder_path="Tensorists/",
    ):
        self.data = data
        self.processor = processor
        self.max_crops = max_crops
        self.num_tokens = num_tokens
        self.base_folder_path = base_folder_path
        self.class_artifact = {
            "airplane": [
                "Artificial noise patterns in uniform surfaces",
                "Metallic surface artifacts",
                "Impossible mechanical connections",
                "Inconsistent scale of mechanical parts",
                "Physically impossible structural elements",
                "Implausible aerodynamic structures",
                "Misaligned body panels",
                "Impossible mechanical joints",
                "Distorted window reflections",
            ],
            "automobile": [
                "Artificial noise patterns in uniform surfaces",
                "Metallic surface artifacts",
                "Impossible mechanical connections",
                "Inconsistent scale of mechanical parts",
                "Physically impossible structural elements",
                "Incorrect wheel geometry",
                "Misaligned body panels",
                "Impossible mechanical joints",
                "Distorted window reflections",
            ],
            "ship": [
                "Artificial noise patterns in uniform surfaces",
                "Metallic surface artifacts",
                "Impossible mechanical connections",
                "Inconsistent scale of mechanical parts",
                "Physically impossible structural elements",
                "Misaligned body panels",
            ],
            "truck": [
                "Artificial noise patterns in uniform surfaces",
                "Metallic surface artifacts",
                "Impossible mechanical connections",
                "Inconsistent scale of mechanical parts",
                "Physically impossible structural elements",
                "Incorrect wheel geometry",
                "Misaligned body panels",
                "Impossible mechanical joints",
                "Distorted window reflections",
            ],
            "bird": [
                "Unrealistic eye reflections",
                "Misshapen ears or appendages",
                "Anatomically impossible joint configurations",
                "Unnatural pose artifacts",
                "Biological asymmetry errors",
                "Regular grid-like artifacts in textures",
                "Impossible foreshortening in animal bodies",
                "Misaligned bilateral elements in animal faces",
                "Over-smoothing of natural textures",
            ],
            "cat": [
                "Unrealistic eye reflections",
                "Misshapen ears or appendages",
                "Anatomically impossible joint configurations",
                "Unnatural pose artifacts",
                "Biological asymmetry errors",
                "Regular grid-like artifacts in textures",
                "Impossible foreshortening in animal bodies",
                "Misaligned bilateral elements in animal faces",
                "Over-smoothing of natural textures",
                "Anatomically incorrect paw structures",
                "Improper fur direction flows",
            ],
            "deer": [
                "Unrealistic eye reflections",
                "Misshapen ears or appendages",
                "Anatomically impossible joint configurations",
                "Unnatural pose artifacts",
                "Biological asymmetry errors",
                "Regular grid-like artifacts in textures",
                "Impossible foreshortening in animal bodies",
                "Misaligned bilateral elements in animal faces",
                "Over-smoothing of natural textures",
                "Improper fur direction flows",
            ],
            "dog": [
                "Unrealistic eye reflections",
                "Misshapen ears or appendages",
                "Anatomically impossible joint configurations",
                "Unnatural pose artifacts",
                "Biological asymmetry errors",
                "Regular grid-like artifacts in textures",
                "Impossible foreshortening in animal bodies",
                "Misaligned bilateral elements in animal faces",
                "Over-smoothing of natural textures",
                "Dental anomalies in mammals",
                "Anatomically incorrect paw structures",
                "Improper fur direction flows",
            ],
            "frog": [
                "Unrealistic eye reflections",
                "Misshapen ears or appendages",
                "Anatomically impossible joint configurations",
                "Unnatural pose artifacts",
                "Biological asymmetry errors",
                "Regular grid-like artifacts in textures",
                "Impossible foreshortening in animal bodies",
                "Misaligned bilateral elements in animal faces",
                "Over-smoothing of natural textures",
            ],
            "horse": [
                "Unrealistic eye reflections",
                "Misshapen ears or appendages",
                "Anatomically impossible joint configurations",
                "Unnatural pose artifacts",
                "Biological asymmetry errors",
                "Regular grid-like artifacts in textures",
                "Impossible foreshortening in animal bodies",
                "Misaligned bilateral elements in animal faces",
                "Over-smoothing of natural textures",
                "Dental anomalies in mammals",
            ],
            "Major Issues": [
                "Discontinuous surfaces",
                "Non-manifold geometries in rigid structures",
                "Asymmetric features in naturally symmetric objects",
                "Texture bleeding between adjacent regions",
                "Excessive sharpness in certain image regions",
                "Artificial smoothness",
                "Movie-poster-like composition of ordinary scenes",
                "Unnatural lighting gradients",
                "Fake depth of field",
                "Abruptly cut-off objects",
                "Color coherence breaks",
                "Spatial relationship errors",
                "Depth perception anomalies",
                "Over-sharpening artifacts",
                "Incorrect reflection mapping" "Inconsistent object boundaries",
                "Floating or disconnected components",
                "Texture repetition patterns",
                "Unrealistic specular highlights",
                "Inconsistent material properties",
                "Inconsistent shadow directions",
                "Multiple light source conflicts",
                "Missing ambient occlusion",
                "Incorrect perspective rendering",
                "Scale inconsistencies within single objects",
                "Aliasing along high-contrast edges",
                "Blurred boundaries in fine details",
                "Jagged edges in curved structures",
                "Random noise patterns in detailed areas",
                "Loss of fine detail in complex structures",
                "Artificial enhancement artifacts",
                "Repeated element patterns",
                "Systematic color distribution anomalies",
                "Frequency domain signatures",
                "Unnatural color transitions",
                "Resolution inconsistencies within regions",
                "Glow or light bleed around object boundaries",
                "Ghosting effects: Semi-transparent duplicates of elements",
                "Cinematization effects",
                "Dramatic lighting that defies natural physics",
                "Artificial depth of field in object presentation",
                "Unnaturally glossy surfaces",
                "Synthetic material appearance",
                "Multiple inconsistent shadow sources",
                "Exaggerated characteristic features",
                "Scale inconsistencies within the same object class"
                "Incorrect skin tones",
            ],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = self.data["label"][idx]
        respnse = self.data["response"][idx]
        question = f"""Analyze the provided image to identify and explain distinguishing artifacts that indicate it is fake. Provide clear, concise explanations (maximum 50 words each) using the specified artifacts below. Include positional references like 'top left' or 'bottom right' when relevant. DO NOT include any other sentences or artifacts in your response.\nExample output: \nResolution and Detail: Missing fine details like individual hairs, replaced by smooth texture\nAnatomical Inconsistencies: Distorted leg proportions and joint angles\nONLY use the artifacts listed below: \n {self.class_artifact[label]+self.class_artifact['Major Issues']}"""

        prompt = prompt = f"""<|im_start|>user
                <image>
                {question}<|im_end|>
                <|im_start|>assistant \n
                {respnse}
                """
        # print(prompt)

        image_path = self.base_folder_path + self.data["image_path"][idx]
        raw_image = Image.open(image_path)
        inputs = processor(
            prompt,
            [raw_image],
            model,
            max_crops=self.max_crops,
            num_tokens=self.num_tokens,
            return_tensors="pt",
            max_length=768,
            padding="max_length",
            truncation=True,
        )
        # inputs['input_ids'] = inputs['input_ids']
        # inputs['attention_mask'] = inputs['attention_mask']

        output = {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "image_features": inputs["image_features"].squeeze(),
            "labels": inputs["input_ids"].squeeze(),
        }

        return output


def fixed_cross_entropy(
    source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = torch.nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction=reduction
    )
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(
        shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    )
    return loss


outputs = pd.read_csv("Tensorists/output_final.csv")
outputs.head()

dataset = MyDataset(outputs, processor)

# Hyperparameters
gradient_accumulation_steps = 1
batch_size = 2
epochs = 3
steps_per_epoch = ((len(dataset)) // batch_size) // gradient_accumulation_steps
total_steps = steps_per_epoch * epochs
warmup_steps = 20
previous_loss = np.inf
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
model = model.to(device)

checkpoint_dir = "Tensorists/checkpoints_ft_mc_llava"

# Check if directory exists
if not os.path.exists(checkpoint_dir):
    # Create directory
    os.makedirs(checkpoint_dir)

global_step = 0
for epoch in range(epochs):
    total_loss = 0
    i = 0
    torch.cuda.empty_cache()
    epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in train_loader:
        i += 1
        global_step += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        # print(batch['input_ids'].shape)
        # print(batch['attention_mask'].shape)
        # print(batch['image_features'].shape)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image_features=batch["image_features"],
            labels=batch["labels"],
        )

        loss = out.loss
        loss.backward()
        # accelerator.backward(loss)

        total_loss += loss.item()

        if i % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if global_step % int(500) == 0:
            if (total_loss / i) < previous_loss:
                checkpoint_path = (
                    f"Tensorists/checkpoints_ft_mc_llava/checkpoint_{global_step}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                # accelerator.save_model(unwrapped_model,checkpoint_path)
                print(f"Saved checkpoint at step {global_step}")
                previous_loss = total_loss / i
