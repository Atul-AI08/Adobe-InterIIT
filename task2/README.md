## Overview

This repository provides a collection of pre-configured models and training scripts to work with multiple **Vision-Language (VL) models** using **LoRA adapters** for efficient transfer learning. The following models are included:

- **Qwen2-VL-7B-Instruct**
- **Llama-3.2-11B-Vision-Instruct**
- **Paligemma-3B**
- **MC-LLaVA-3b**

You can fine-tune these models using a custom dataset, and utilize inference scripts to run them for various tasks such as image captioning, object recognition, and multi-modal interactions.

## Prerequisites

To use this repository, make sure you have the following installed:

- Python 3.10 or later
- PyTorch
- Hugging Face Transformers library
- LLaMA factory


Ensure that you have access to a suitable GPU environment to train and run these models efficiently.

## Repo Structure

```
.
├── LLaMA-Factory-main/
│   ├── train_qwen2vl.json      # For training qewen2 model
│   ├── train_llama.json        # For training llama model
│   ├── train_paligemma.json    # For training paligemma model
│   ├── merge_qwen2vl.json      # For merging qewen2 model
│   ├── merge_llama.json        # For merging llama model
│   ├── merge_paligemma.json    # For merging paligemma model
│   └──data/
|       └── dataset_info.json   # Custom dataset for training
├───finetune_mc_llava.py        # For fine-tuning mc-llava model
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

---

All necessary dependencies are listed in the requirements.txt file. Requirements needed for LLama-Factory can be found in the 'LLaMA-Factory-main/requirements.txt' file.
Please ensure they are installed before proceeding.