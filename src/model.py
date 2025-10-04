import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import inspect
import math


@dataclass
class GPTConfig:
    context_length: int = 1024    # max context / sequence length
    vocab_size: int = 50257    # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    num_layers: int = 12
    embd_size: int = 768    # embedding dim
    num_heads: int = 12
    num_kv_heads: int = None  # for GQA, if None then uses MHA
    rope_theta: float = 10000.0  # RoPE base frequency


class RMSNorm(nn.Module):
    """RMSNorm: Root Mean Square Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Reshape freqs_cis to match the shape of xq_ and xk_
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.embd_size = config.embd_size
        self.head_dim = config.embd_size // config.num_heads
        
        # GQA: if num_kv_heads is specified, use grouped query attention
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads
        self.num_kv_groups = config.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 1
        
        assert config.embd_size % config.num_heads == 0, f"embedding dim should be divisible by number of heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        # Key, query, value projections with GQA
        self.c_attn = nn.Linear(config.embd_size, config.embd_size + 2 * self.num_kv_heads * self.head_dim)
        self.c_proj = nn.Linear(config.embd_size, config.embd_size)
        self.c_proj.SCALE_INIT = 1.0
        
        # Precompute rotary embeddings
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                self.head_dim,
                config.context_length,
                config.rope_theta,
            ),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        
        # Calculate query, key, values for all heads
        qkv = self.c_attn(x)
        
        # Split into query, key, value with GQA
        q, k, v = torch.split(
            qkv, 
            [self.embd_size, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim], 
            dim=-1
        )
        
        # Reshape queries
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Reshape keys and values for KV heads
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings to queries and keys
        freqs_cis = self.freqs_cis[:T]
        q, k = apply_rotary_emb(q, k, freqs_cis)
        
        # For GQA, repeat K and V if num_kv_heads < num_heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Use flash attention with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.embd_size, 4 * config.embd_size)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.embd_size, config.embd_size)
        self.c_proj.SCALE_INIT = 1.0

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """Transformer block with RMSNorm"""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.embd_size)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.embd_size)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Remove positional embeddings since we're using RoPE
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.embd_size),
            h = nn.ModuleList([Block(self.config) for _ in range(self.config.num_layers)]),
            ln_f = RMSNorm(self.config.embd_size)
        ))
        
        # language modeling head
        self.lm_head = nn.Linear(self.config.embd_size, self.config.vocab_size, bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std /= (2 * self.config.num_layers)**0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.context_length, f'sequence length {T} should be <= {self.config.context_length}'
        
        # Only token embeddings since we're using RoPE
        x = self.transformer.wte(idx)  # (B, T, embd_size)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2': dict(num_layers=12, num_heads=12, embd_size=768),
            'gpt2-medium': dict(num_layers=24, num_heads=16, embd_size=1024),
            'gpt2-large': dict(num_layers=36, num_heads=20, embd_size=1280),
            'gpt2-xl': dict(num_layers=48, num_heads=25, embd_size=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['context_length'] = 1024
        config_args['num_kv_heads'] = None  # Standard MHA for GPT-2
        config_args['rope_theta'] = 10000.0

        # Create a from-scratch minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()

        # Init a huggingface transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        # Filter out keys we don't need
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        
        # Remove positional embedding keys since we're using RoPE
        sd_keys_hf = [k for k in sd_keys_hf if 'wpe' not in k]
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Copy weights while handling differences in architecture
        for k in sd_keys_hf:
            if k not in sd:
                print(f"Skipping {k} (not in target model)")
                continue
                
            if any(k.endswith(w) for w in transposed):
                # need to transpose Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}: {sd_hf[k].shape} vs {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        print("Note: Positional embeddings (RoPE) are initialized from scratch")
        return model

    def configure_optimizers(self, weight_decay, lr, device_type, master_process):
        """
        Configure optimizer with weight decay
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # create optim groups
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
        
        # use fused version of AdamW optimizer if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if master_process:
            print(f'using fused AdamW optimizer: {use_fused}')
            
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
