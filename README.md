# Adobe-InterIIT

## Project Overview

This project addresses two primary tasks related to image classification and adversarial robustness, along with explainability of AI-generated images. The repository is structured into three main folders: **Task1**, **Attacks**, and **Task2**.

---

## Repository Structure

### **1. Task1 Folder**
This folder contains implementations of various methods to classify an image as **real** or **fake**.

#### Features:
- Multiple classification algorithms to identify AI-generated images.
- Configurable pipelines for training, evaluation, and testing.
- Comprehensive metrics to evaluate model performance, including accuracy, precision, and recall.

### **2. Attacks Folder**
This folder focuses on adversarial attacks designed to test and inhibit the robustness of models developed in **Task1**. The attacks aim to expose vulnerabilities in classification models by manipulating input images.

#### Features:
- Scripts for generating adversarial examples using various attack strategies.

### **3. Task2 Folder**
This folder deals with explaining why certain images are classified as **fake**. The explanation process leverages **Vision-Language Models (VLMs)** fine-tuned on resized images to provide insights into the artifacts and patterns that led to classification.

#### Features:
- Scripts for fine-tuning VLMs on task-specific data.
- Inference pipelines for generating explainability outputs.
- Tools to visualize and interpret artifacts present in **fake** images.

---

## Implementation Details
- Refer to the individual folders (`task1`, `Attacks`, `task2`) for detailed implementation documentation, scripts, and configuration options.
- Each folder contains a `README.md` file with specific instructions for its respective code.

---