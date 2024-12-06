## Overview

This repository contain various models and approaches that we tried for task 1 i.e the classification task:

- **DIRE**
- **AEROBLADE**
- **RIGID**
- **RESNET50**
- **CLIP**
- **Robust CLIP**
- **VIT**
- **TinyVIT**


All files can be run as-is, with only minor adjustments needed for file paths and directory locations.

## Prerequisites

To use this repository, make sure you have the following installed:

- Python 3.7 or later
- All necessary dependencies are listed in the requirements.txt file. Please ensure they are installed before proceeding.


Ensure that you have access to a suitable GPU environment to train and run these models efficiently.

## Repo Structure

```
.
├── vae-fft/
│   ├── Attention.py               # for self-attention, multi-headed attention and cross-attention 
│   ├── EarlyStopping.py           # for early stopping
│   ├── Encoder.py                 # for variational auto-encoder implementation
│   ├── model.py                   # for model architecture   
│   ├── train.py                   # for train and test loops
│   ├── utils.py                   # for patch embeddings and fft transform
│
├───AerobladeClassifier.ipynb      # Implementation of AEROBLADE training-free classifier using autoencoders
├───ClipClassifier.ipynb           # Implementation and training of CLIP + NN classifier
├───DIRE.ipynb                     # Implementation of DIRE training-free classifier using noising-denoising
├───RIGID.ipynb                    # Implementation of RIGID training-free classifier using cosine similarity
├───Robust_Clip.ipynb              # Finetuning RobustCLIP with freezed weights and added dense layers
├───ViT_Finetune.ipynb             # Finetuing of Google Base ViT for 2 epochs 
├───output_tinyvit.py              # Inference on tinyvit
├───train_resnet.py                # Finetuning of Resnet50 with last 10 layers frozen and added fully connected layers
├───train_tinyvit.py               # Finetuning of tinyvit for 10 epochs

```
