import sys
import math
import numpy as np
import time
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        # key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1.0
        # not really a bias, more of a mask, but following OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)    # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)    # (B,T,C), (B,T,C), (B,T,C)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)    # (B,nh,T,hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)    # (B,nh,T,hs)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)    # (B,nh,T,hs)

        # attn = q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1])    # (B,nh,T,hs) @ (B,nh,hs,T) --> (B,nh,T,T)
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # out = attn @ v    # (B,nh,T,T) @ (B,nh,T,hs) --> (B,nh,T,hs)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)    # flash-attention paper (significantly faster)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_blocks: int = 12
    n_embd: int = 768
    n_heads: int = 12


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.n_blocks)]),
            ln_f = nn.LayerNorm(self.config.n_embd)
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        # weight sharing scheme (reduces 768*50267=~40M params, fewer params, more efficient)
        self.transformer.wte.weight = self.lm_head.weight

        # init params (iterates over all submodules and applies _init_weights)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std /= (2 * self.config.n_blocks)**0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)    # as per openai gpt-2 source code
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f'sequence length {T} should be less than {self.config.block_size}'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)    # (T,)
        pos_embd = self.transformer.wpe(pos)    # (T, n_embd)
        tok_embd = self.transformer.wte(idx)    # (B, T, n_embd)
        x = pos_embd + tok_embd    # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)    # (B, T, n_embd)
        logits = self.lm_head(x)    # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT2 model weights from huggingface """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2': dict(n_blocks=12, n_heads=12, n_embd=768),    # 124M params
            'gpt2-medium': dict(n_blocks=24, n_heads=16, n_embd=1024),    # 350M params
            'gpt2-large': dict(n_blocks=36, n_heads=20, n_embd=1280),    # 774M params
            'gpt2-xl': dict(n_blocks=48, n_heads=25, n_embd=1600),    # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024

        # create a from-scratch minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # init a huggingface transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(sd_keys) == len(sd_keys_hf), f"mismatched keys {len(sd_keys)} != {len(sd_keys_hf)}"

        # copy while ensuring all parameters are aligned in names and shape
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # need to transpose Conv1D weights
                assert sd_hf[k].shape[::-1]  == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, lr, device):
        # start with all of the candidate params (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. any parameters that are 2D will be weight decayed, otherwise no.
        # i.e., all weight tensors in matuls + embeddings will decay, biases and layernorms don't



n_return_sequences = 5
max_length = 30

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'Using device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

class DataLoaderLite:
    def __init__(self, B, T):
        self.B, self.T = B, T
        with open('../data/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'Loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')
        self.curr_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        batch = self.tokens[self.curr_pos:self.curr_pos + B*T + 1]
        x_batch = batch[:-1].view(B, T)
        y_batch = batch[1:].view(B, T)
        self.curr_pos += B*T
        if self.curr_pos + B*T + 1 > len(self.tokens):
            self.curr_pos = 0
        return x_batch, y_batch


# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
print('Success')
model.eval()
model = model.to(device)

# use compile almost always unless debugging (makes training faster)
# model = torch.compile(model)

train_loader = DataLoaderLite(B=16, T=32)

# enable TF32 precision
# torch.set_float32_matmul_precision('high')

# print(logits.shape)
# print(loss)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(step):
    # 1) linear warmup for warmup_iters steps
    if step < warmup_steps:
        return max_lr * (step+1) / warmup_steps
    # 2) if step > lr_decay_iters, return min lr
    if step > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
for step in range(max_steps):
    t0 = time.time()
    inp, tar = train_loader.next_batch()
    inp, tar = inp.to(device), tar.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(inp, tar)
    # import code; code.interact(local=locals())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set lr for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups():
        param_group['lr'] = lr
    optimizer.step()
    # torch.cuda.synchronize()    # wait for the GPU to finish work
    dt = (time.time() - t0) * 1000    # in ms
    tokens_processed = train_loader.B * train_loader.T
    tokens_per_sec = tokens_processed / dt
    print(f'step {step:4d} | loss: {loss.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec: {tokens_per_sec}')

sys.exit(0)

tokens = enc.encode('hello I am a language model,')
tokens = torch.tensor(tokens, dtype=torch.long)    # (8)
tokens = tokens.unsqueeze(0).repeat(n_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

while x.shape[-1] <= max_length:
    with torch.no_grad():
        logits = model(x)    # (B, T, vocab_size)
        logits = logits[:, -1, :]    # (B, vocab_size)
        probs = F.softmax(logits, dim=-1)    # (B, vocab_size)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # (B,50), (B,50)
        ix = torch.multinomial(topk_probs, num_samples=1)    # (B,1)
        next_tok = torch.gather(topk_indices, -1, ix)    # (B,1)
        x = torch.cat([x, next_tok], dim=1)

print(x.shape)

for i in range(n_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)