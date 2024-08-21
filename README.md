This work tries to reproduce the GPT-2 paper implementation and trained on FineWebEdu dataset. It has been adapted from Andrej Karpathy's tutorial.

This is a simplified PyTorch implementation of the GPT-2 model. The goal of this project is to provide a simple and easy-to-understand implementation. The code is not optimized for speed and is not intended to be used for production.

## Usage

Dependencies:
- PyTorch 1.13.1 (install instructions)
- torchvision 0.14.1 (install instructions)
- matplotlib 3.7.1 to generate plots for model inspection

Run the below script to install the dependencies
```bash
pip install -r requirements.txt
```

You can find the implementation in the `gpt_model.py` file. The main class is `GPT`, which contains the embedding layer, the transformer encoder, and the classification head. All of the modules are heavily commented to make it easier to understand.

The model config is defined as a python dictionary in `train.py`, you can experiment with different hyperparameters there. Training parameters can be passed using the command line. For example, to train the model for 10 epochs with a batch size of 32, you can run:

```bash
python train.py --exp-name vit-with-10-epochs --epochs 10 --batch-size 32
```

Please have a look at the `train.py` file for more details.

## Dataset

GPT2 has been trained on WebText dataset (official version not released). 

The model has been trained on FineWeb-Edu dataset (high quality educational content), a subset of FineWeb (filtered version of high-quality CommonCrawl dataset). It contains 1.3 trillion and 5.4 trillion high-educational content gpt2 tokens. We have used a subset sample-10BT (10 billion) tokens subset, sufficient enough to get close to gpt2 performance and simple to work with. 


How to train the model? 

```
python3 gpt2_dev.py
```

The model is trained (from scratch) using multiple GPUs using PyTorch's DDP. 

Model Architecture:

The model has a token embedding layer followed by a position embedding table. The outputs of these layers goes into a series of blocks. Each block consists of layernorm followed by multiheaded self-attention layer, and MLP with residual connections. 

The model is trained to predict next token given the past tokens. 


## References:
1. GPT2 paper
2. GPT3 paper
3. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré; FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
4. Andrej Karpathy video link