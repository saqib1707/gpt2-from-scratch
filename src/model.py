import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 'embd_size' sized vector divided into 'num_heads' heads
        assert config.embd_size % config.num_heads == 0, f"embedding dim should be divisible by number of heads"
        self.num_heads = config.num_heads
        self.embd_size = config.embd_size
        # batched key, query, and value projections for all heads
        self.c_attn = nn.Linear(config.embd_size, 3 * config.embd_size)
        self.c_proj = nn.Linear(config.embd_size, config.embd_size)
        self.c_proj.SCALE_INIT = 1.0
        # not really a bias, more of a mask, but following OpenAI/HF naming convention
        # self.register_buffer("bias", torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, T, C = x.shape
        # calculate query, key, values for all heads in a batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels
        qkv = self.c_attn(x)    # (B, T, 3C)
        q, k, v = qkv.split(self.embd_size, dim=-1)    # (B,T,C), (B,T,C), (B,T,C)
        q = q.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(1, 2)    # (B,nh,T,hs)
        k = k.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(1, 2)    # (B,nh,T,hs)
        v = v.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(1, 2)    # (B,nh,T,hs)
        # attn = q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1])    # (B,nh,T,hs) @ (B,nh,hs,T) --> (B,nh,T,T)
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # out = attn @ v    # (B,nh,T,T) @ (B,nh,T,hs) --> (B,nh,T,hs)
        # flash-attention paper (significantly faster, but logically the same as above 4 lines)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)    # (B,nh,T,hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)    # (B,nh,T,hs) --> (B,T,nh,hs) --> (B,T,C=nh*hs)
        out = self.c_proj(out)    # (B,T,C) --> (B,T,C)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embd_size, 4 * config.embd_size)
        self.gelu = nn.GELU(approximate='tanh')    # approximate='tanh' used to try to reproduce gpt2 paper
        self.c_proj = nn.Linear(4 * config.embd_size, config.embd_size)
        self.c_proj.SCALE_INIT = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """ Transformer Encoder block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embd_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embd_size)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.embd_size),
            wpe = nn.Embedding(self.config.context_length, self.config.embd_size),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.num_layers)]),
            ln_f = nn.LayerNorm(self.config.embd_size)
        ))
        # language modeling head
        self.lm_head = nn.Linear(self.config.embd_size, self.config.vocab_size, bias=False)
        # weight sharing scheme (reduces 768*50267=~40M params, fewer params, more efficient)
        self.transformer.wte.weight = self.lm_head.weight
        # init params (iterates over all submodules and applies _init_weights)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std /= (2 * self.config.num_layers)**0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)    # as per openai gpt-2 source code
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.context_length, f'sequence length {T} should be <= {self.config.context_length}'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)    # (T,)
        pos_embd = self.transformer.wpe(pos)    # (T, embd_size)
        tok_embd = self.transformer.wte(idx)    # (B, T, embd_size)
        x = pos_embd + tok_embd    # (B, T, embd_size)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)    # (B, T, embd_size)
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
            'gpt2': dict(num_layers=12, num_heads=12, embd_size=768),    # 124M params
            'gpt2-medium': dict(num_layers=24, num_heads=16, embd_size=1024),    # 350M params
            'gpt2-large': dict(num_layers=36, num_heads=20, embd_size=1280),    # 774M params
            'gpt2-xl': dict(num_layers=48, num_heads=25, embd_size=1600),    # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['context_length'] = 1024

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

    def configure_optimizers(self, weight_decay, lr, device_type, master_process):
        """
        Essentially implements weight decay (regularization tool, by decaying the weights, we 
        forcing the optimizer to use more of the weights, and not allowing any single weight to dominate)
        """
        # start with all of the candidate params (that require gradient)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # create optim groups: any parameters that are 2D will be weight decayed, otherwise no.
        # i.e., all weight tensors in matmuls + embeddings will decay, whereas biases and layernorms won't be decayed
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f'num decay parameter tensors: {len(decay_params)} with {num_decay_params:,} parameters')
            print(f'num nodecay parameter tensors: {len(nodecay_params)} with {num_nodecay_params:,} parameters')
        
        # use fused version of AdamW optimizer (faster than non-fused version)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            print(f'using fused AdamW optimizer: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer