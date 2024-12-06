## Overview

This repository contain various models that we tried for our experimentation. Our experimentation includes the following model:

- **DIRE**
- **AEROBLADE**
- **RIGID**
- **RESNET50**
- **CLIP**
- **Robust CLIP**
- **VIT**
- **TinyVIT**


You can fine-tune these models using a custom dataset, and utilize inference scripts to run them for various tasks such as image captioning, object recognition, and multi-modal interactions.

## Prerequisites

To use this repository, make sure you have the following installed:

- Python 3.7 or later
- PyTorch (version >= 1.10)
- Hugging Face Transformers library
- LLaMA factory


Ensure that you have access to a suitable GPU environment to train and run these models efficiently.

## Repo Structure

```
.
├── LLaMA-Factory-main/
│   ├── train_qwen2vl.json      # for training qewen2 model
│   ├── train_llama.json        # for training llama model
│   ├── train_paligemma.json    # for training paligemma model
│   ├── merge_qwen2vl.json      # for merging qewen2 model
│   ├── merge_llama.json        # for merging llama model
│   ├── merge_paligemma.json    # for merging paligemma model
│   └──data/
|       └── dataset_info.json   # Custom dataset for training
├───finetune_mc_llava.py        # for fine-tuning mc-llava model
```

### Configuration Files

The configurations for each model are located in the train config json files. These configurations contain:

- Training hyperparameters
- LoRA adapter settings

### Dataset File

- `train.json`: This file is used to configure and load your custom dataset for training. It should be structured to include pairs of images and text for vision-language tasks.

Example format for `train.json`:

```json
{
    "messages": [
        {
            "content": "<image><image>{prompt}",
            "role": "user",
        },
        {
            "content": "{response}",
            "role": "assistant"
        }
    ],
    "images": ["path/to/val_image1.jpg", "path/to/val_image2.jpg", ...],
}
```

## Training the Models

You can fine-tune the models using the `train.py` script, specifying the model configuration, dataset, and model output path.

### To train  model:

```bash
cd LLaMA-Factory-main
llamafactory-cli train train_qwen2vl.json
```

### To train the **MCLLava** model:

```bash
python finetune_mc_llava.py
```

## Merging LoRA Adapters

To merge the LoRA adapter with a base model, run the following script. This merges the LoRA adapter weights into the base model.

```bash
llamafactory-cli export merge_qwen2vl.json
```

This will merge the LoRA adapter weights into the model and save the merged model.