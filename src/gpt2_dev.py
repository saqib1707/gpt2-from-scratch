import os
import sys
import math
import numpy as np
import time
from dataclasses import dataclass
import inspect
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# import code; code.interact(local=locals())
from hellaswag import render_example, iterate_examples, get_most_likely_row


script_dir = os.path.dirname(__file__)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 'n_embd' sized vector divided into 'n_heads' heads
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        # key, query, value projections for all heads but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_INIT = 1.0
        # not really a bias, more of a mask, but following OpenAI/HF naming convention
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels
        qkv = self.c_attn(x)    # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=-1)    # (B,T,C), (B,T,C), (B,T,C)
        q = q.view(B, T, self.n_heads, self.n_embd // self.n_heads).transpose(1, 2)    # (B,nh,T,hs)
        k = k.view(B, T, self.n_heads, self.n_embd // self.n_heads).transpose(1, 2)    # (B,nh,T,hs)
        v = v.view(B, T, self.n_heads, self.n_embd // self.n_heads).transpose(1, 2)    # (B,nh,T,hs)
        # attn = q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1])    # (B,nh,T,hs) @ (B,nh,hs,T) --> (B,nh,T,T)
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # out = attn @ v    # (B,nh,T,T) @ (B,nh,T,hs) --> (B,nh,T,hs)
        # flash-attention paper (significantly faster, but logically same as above 4 lines)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')    # approximate='tanh' used to try to reproduce gpt2 paper
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
    block_size: int = 1024    # max context length
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

    def configure_optimizers(self, weight_decay, lr, device_type):
        # start with all of the candidate params (that require gradient)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        # create optim groups. any parameters that are 2D will be weight decayed, otherwise no.
        # i.e., all weight tensors in matmuls + embeddings will decay, biases and layernorms won't be decayed
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        n_decay_params = sum(p.numel() for p in decay_params)
        n_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f'num decay parameter tensors: {len(decay_params)} with {n_decay_params} parameters')
            print(f'num nodecay parameter tensors: {len(nodecay_params)} with {n_nodecay_params} parameters')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            print(f'Using fused AdamW optimizer: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer


def estimate_lr(step, warmup_steps, max_steps, max_lr, min_lr):
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


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split='train'):
        self.B, self.T = B, T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}
        # get the shard filenames
        data_root = os.path.join(script_dir, "../data/edu_fineweb10B")
        shard_filenames = os.listdir(data_root)
        shard_filenames = sorted([filename for filename in shard_filenames if split in filename])
        self.shard_filepaths = [os.path.join(data_root, filename) for filename in shard_filenames]
        assert len(self.shard_filepaths) > 0, f'no shards found for split {split}'
        if master_process:
            print(f'found {len(self.shard_filepaths)} shards for split {split}')
        self.reset()

    def load_tokens(self, filepath):
        tokens = torch.tensor(np.load(filepath).astype(np.int32), dtype=torch.long)
        return tokens

    def reset(self):
        # state, init at shard 0
        self.curr_shard = 0
        self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
        self.curr_pos = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        batch = self.tokens[self.curr_pos : self.curr_pos + B*T + 1]
        x_batch = batch[:-1].view(B, T)
        y_batch = batch[1:].view(B, T)
        self.curr_pos += B * T * self.num_processes
        if self.curr_pos + (B * T + 1) > len(self.tokens):
            self.curr_shard = (self.curr_shard + 1) % len(self.shard_filepaths)
            self.tokens = self.load_tokens(self.shard_filepaths[self.curr_shard])
            self.curr_pos = self.B * self.T * self.process_rank
        return x_batch, y_batch


# set up DDP (distributed data parallel)
# 'torchrun' command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
# RANK and LOCAL_RANK same for (single node, multi-GPU) settings, may differ for (multinode, 
# multi GPU) settings. 
ddp = int(os.environ.get('RANK', -1)) != -1    # if this is a ddp run
if ddp:
    # use of ddp requires CUDA
    assert torch.cuda.is_available(), f'use of DDP requires CUDA'
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0    # this process will do logging, checkpointing, etc.
else:
    # not using ddp
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # for apple macbook GPUs
        device = 'mps'
    print(f'Using device: {device}')

device_type = 'cuda' if device.startswith('cuda') else 'cpu'

# set seed for reproducibility
torch.manual_seed(1337)    # sets seed for random number generation on CPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)    # sets seed for random number generation on GPU

total_batch_size = 524288    # =2^19 tokens/step update, ~0.5M tokens (used in openai gpt3 paper)
mini_batch_size = 32    # mini batch size
block_size = 1024    # max sequence length
assert total_batch_size % (mini_batch_size * block_size * ddp_world_size) == 0, 'ensure total_batch_size divisible by B*T*ddp_world_size'
grad_accum_steps = total_batch_size // (mini_batch_size * block_size * ddp_world_size)
if master_process:
    print(f'desired batch size: {total_batch_size}')
    print(f'calculated gradient accumulation steps: {grad_accum_steps}')
print(f'I am GPU: {ddp_rank}, {ddp_local_rank}')

train_loader = DataLoaderLite(B=mini_batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DataLoaderLite(B=mini_batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# enable TF32 precision
torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))    # 50304 used instead of 50257
# model = GPT.from_pretrained('gpt2')    # init from OpenAI GPT-2
model.to(device)    # move model to device
use_compile = False
if use_compile:
    model = torch.compile(model)    # use torch compile almost always unless debugging (makes training faster)

if ddp:
    # wraps the model in DDP container (forward pass is unchanged, but after backward pass,
    # gradients computed across each processes averaged by DDP using 'AllReduce' and shared across 
    # all processes so that each process has same gradients)
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

max_lr = 6e-4 * 2
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 * 1   # 10^10 / 2^19 ~ 19073 for 1 epoch
eval_every = 250

optimizer = raw_model.configure_optimizers(weight_decay=0.1, lr=max_lr, device_type=device_type)
token_encoder = tiktoken.get_encoding('gpt2')

# create the log directory
logdir = 'logs'
os.makedirs(logdir, exist_ok=True)
logpath = os.path.join(logdir, 'log.txt')
with open(logpath, 'w') as f:
    pass

start_time = time.time()
for step in range(max_steps):
    t0 = time.time()
    is_last_step = (step == max_steps - 1)

    # evaluate validation loss
    if step % eval_every == 0 or is_last_step:
        model.eval()    # sets model to eval mode
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0
            val_steps = 20
            for _ in range(val_steps):
                inp, tar = val_loader.next_batch()
                inp, tar = inp.to(device), tar.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(inp, tar)
                loss /= val_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f'Val loss: {val_loss_accum.item():.4f}')
            with open(logpath, 'a') as f:
                f.write(f'{step} val {val_loss_accum.item():.4f}\n')
            if step > 0 and (step % 10000 == 0 or is_last_step):
                ckpt_path = os.path.join(logdir, f'model_{step:05d}.pt')
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }    # add optimizer.state_dict(), rng_seeds, etc. if resuming training
                torch.save(checkpoint, ckpt_path)

    # evaluate HellaSwag every once in a while
    if ((step > 0 and step % eval_every == 0) or is_last_step) and (not use_compile):
        n_total = 0
        n_correct_norm = 0
        for i, example in enumerate(iterate_examples('val')):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)    # (4,N), (4,N), (4,N)
            tokens, mask = tokens.to(device), mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            n_total += 1
            n_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            n_total = torch.tensor(n_total, device=device, dtype=torch.long)
            n_correct_norm = torch.tensor(n_correct_norm, device=device, dtype=torch.long)
            dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_correct_norm, op=dist.ReduceOp.SUM)
            n_total = n_total.item()
            n_correct_norm = n_correct_norm.item()
        acc_norm = n_correct_norm / n_total
        if master_process:
            print(f'HelloSwag accuracy: {n_correct_norm}/{n_total}={acc_norm:.4f}')
            with open(logpath, 'a') as f:
                f.write(f'{step} hella {acc_norm:.4f}\n')

    # generate from the model every once in a while
    if ((step > 0 and step % eval_every == 0) or is_last_step) and (not use_compile):
        model.eval()
        n_return_seq = 4
        max_tokens = 32
        tokens = token_encoder.encode("Hello, I am a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)    # (n,)   n : current sequence length
        tokens = tokens.unsqueeze(0).repeat(n_return_seq, 1)    # (1,n) --> (n_return_seq, n)
        gen_tokens = tokens.to(device)
        # create a different rng generator so as not to impact the global rng state used for training
        sample_rng = torch.Generator(device=device)
        # adding 'ddp_rank' in seeding to generate different tokens for different rank processes
        sample_rng.manual_seed(42 + ddp_rank)
        # generate new tokens one token at a time until the sequence length becomes 'max_tokens'
        while gen_tokens.shape[-1] <= max_tokens:
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(gen_tokens)    # (n_return_seq, n, vocab_size)
                logits = logits[:, -1, :]    # (n_return_seq, vocab_size)
                probs = F.softmax(logits, dim=-1)    # (n_return_seq, vocab_size)
                # take top-k 50 probs
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # (n_return_seq, 50), (n_return_seq, 50)
                # sample a token from top-50 probabilities
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)    # (n_return_seq, 1)
                next_tok = torch.gather(topk_indices, -1, ix)    # (n_return_seq, 1)
                gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)
        # decode generated tokens and print generated text
        for i in range(n_return_seq):
            tokens = gen_tokens[i, :max_tokens].tolist()
            gen_text = token_encoder.decode(tokens)
            print(f"> rank {ddp_rank} sample {i}: {gen_text}")

    # training loop
    model.train()    # sets model to train mode
    optimizer.zero_grad()    # resets all gradients
    batch_loss = 0
    for micro_step in range(grad_accum_steps):
        inp, tar = train_loader.next_batch()
        inp, tar = inp.to(device), tar.to(device)
        if ddp:
            # in the final micro_step, sync and avg all gradients across all processes. used by both forward and backward processes
            # can use 'no_sync()' context manager alternatively. 
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # autocast to bfloat16 for faster compute and memory efficiency
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(inp, tar)
        # loss is scaled to account for gradient accumulation, because the gradients just add
        # on each successive backward() call. Addition of gradients corresponds to SUM in the objective,
        # but we want MEAN instead of a SUM
        loss /= grad_accum_steps
        batch_loss += loss.detach()
        # each process accumulates gradients separately when 'require_backward_grad_sync'=False
        # in the final 'micro_step', 'require_backward_grad_sync' becomes True, therefore 
        # gradients are averaged across all processes and shared among them by loss.backward()
        loss.backward()
    if ddp:
        # 'batch_loss' is outside of DDP container, so need to perform 'all_reduce' to 
        # average out 'batch_loss' across all processes of all ranks. 'batch_loss' tensor exists on all GPUs. 
        # 'all_reduce' averages and deposits the result on all the processes
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = estimate_lr(step, warmup_steps, max_steps, max_lr, min_lr)    # determine and set lr for this iteration
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == 'cuda':
        torch.cuda.synchronize()    # wait for the GPU to finish work
    dt = (time.time() - t0) * 1000.0    # in ms
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f'step {step:4d} | loss: {batch_loss.item():.6f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec: {tokens_per_sec:.4f}')
        with open(logpath, 'a') as f:
            f.write(f'{step} train {batch_loss.item():.6f}\n')

if ddp:
    dist.destroy_process_group()

dt = (time.time() - start_time)
print(f"Total time: {dt:.4f}s")
