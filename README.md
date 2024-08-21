# GPT-2 Implementation and Training on FineWebEdu Dataset

This project aims to reproduce the GPT-2 model from the original paper and fine-tune it on the FineWebEdu dataset. The implementation is adapted from Andrej Karpathy's tutorial and offers a simplified, easy-to-understand PyTorch-based approach. Note that this code is intended primarily for educational purposes and is not optimized for speed or production deployment.

### Project Overview
This repository contains a PyTorch implementation of GPT-2, along with scripts for training and evaluation. The model is trained from scratch on the FineWebEdu dataset — a high-quality subset of CommonCrawl tailored for educational content.

### Features
- **Simplified PyTorch Implementation:** Designed to be accessible and well-commented for ease of understanding.
- **Customizable Training:** Hyperparameters are configurable via a Python dictionary and can be easily modified.
- **Multi-GPU Training Support:** Training can be performed using multiple GPUs via PyTorch Distributed Data Parallel (DDP).

## Repository Structure (main files)
- `src/train.py`: Script for training the GPT-2 model with customizable configurations.
- `src/model.py`: Contains the GPT-2 model implementation, including embedding layers, transformer encoders, and the classification head.
- `requirements.txt`: Dependencies required to run the project.

## Getting Started

### Prerequisites
Ensure you have the following dependencies installed:

- numpy
- pytorch
- tiktoken
- transformers (from huggingface)

You can install all dependencies by running:
```bash
pip install -r requirements.txt
```

### Running the Training Script
To start training the GPT-2 model, you can use the following commands:

You can find the implementation in the `model.py` file. The main class is `GPT`, which contains the embedding layer, the transformer encoder, and the prediction head. All of the modules are heavily commented to make it easier to understand.

The model config is defined as a python dictionary in `train.py`, you can experiment with different hyperparameters there. Training parameters can be passed using the command line. For example, to train the model for 5 epochs with a batch size of 32, you can run:

- Single-GPU Training:
```bash
python train.py --num_epochs=5 
```

- Multi-GPU Training (using DDP):
```bash
torchrun --standalone --nproc_per_node=4 train.py
```

For more details on the training process and how to adjust hyperparameters, please refer to the `train.py` script.


## Dataset

The GPT-2 model was originally trained on the WebText dataset (which hasn’t been officially released). The FineWebEdu dataset used here is a high-quality educational subset of the FineWeb dataset, itself a filtered version of CommonCrawl. Our subset, FineWebEdu-10BT, consists of approximately 10 billion tokens, specifically curated for educational content.

To download and generate the dataset:
```bash
python prepare_dataset.py
```

Training was performed from scratch using multiple GPUs with PyTorch's DDP framework.


The model is trained (from scratch) using multiple GPUs using PyTorch's DDP. 

### Model Architecture
The GPT-2 model consists of the following components:

- **Token Embedding Layer:** Encodes input tokens.
- **Positional Embedding Layer:** Adds positional information to the input sequence.
- **Transformer Blocks:** Each block includes layer normalization, multi-headed self-attention, and an MLP with residual connections.
- **Output Head:** The model is trained to predict the next token based on the previous sequence.

The training objective is to predict the next token given the past tokens in a sequence, enabling the model to generate coherent text.


## References:
- [Language Models are Unsupervised Multitask Learners (GPT-2 Paper)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3 Paper: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [FineWebEdu-10B Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)
- Andrej Karpathy's Video Tutorial on GPT


## Acknowledgments
This implementation is inspired by Andrej Karpathy’s approach to making complex AI concepts more accessible.