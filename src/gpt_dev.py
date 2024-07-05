import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


batch_size = 32
block_size = 8
n_embd = 20
n_blocks = 3
dropout = 0.2
lr = 1e-3
max_itr = 10000
eval_interval = 1000
eval_iters = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

torch.manual_seed(1337)

with open('../data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
vocab_size = len(vocab)
stoi = {ch:i for i, ch in enumerate(vocab)}
itos = {i:ch for ch, i in stoi.items()}
encode_seq = lambda sentence: [stoi[ch] for ch in sentence]
decode_seq = lambda seq: ''.join([itos[i] for i in seq])

print(encode_seq('Hi Saqib'))
print(decode_seq(encode_seq('Hi Saqib')))

data = torch.tensor(encode_seq(text), dtype=torch.long)
split_perc = 0.9
train_size = int(len(data) * split_perc)
train_data = data[:train_size]
val_data = data[train_size:]


def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    samp_idxs = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[t:t+block_size] for t in samp_idxs])
    y = torch.stack([data[t+1:t+block_size+1] for t in samp_idxs])
    x, y = x.to(device), y.to(device)
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B,T,head_size)
        k = self.key(x)    # (B,T,head_size)

        wei = q @ k.transpose(-2,-1) / (self.head_size**0.5)  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)

        v = self.value(x)  # (B,T,head_size)
        out = wei @ v   # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU()
        )
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.net(x)
        out = self.dropout(self.proj(out))
        return out


class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_blocks):
        super().__init__()
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, self.n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, self.n_embd)
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4)
        # self.feed_forward = FeedForward(n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=4) for _ in range(n_blocks)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(self.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C=n_embd)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))  # (T,C=n_embd)
        x = tok_emb + pos_emb    # (B,T,C)
        # x = self.sa_heads(x)   # apply one self-attention head (B,T,C)
        # x = self.feed_forward(x)  # (B,T,C)
        x = self.blocks(x)    # (B,T,C)
        x = self.ln_final(x)   # (B,T,C)
        logits = self.lm_head(x)    # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print(logits, targets)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens=100):
        # gen_text = []
        # gen_text.append(idx.item())
        for _ in range(max_new_tokens):
            # print(idx.shape)
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)    # (B,T,C)
            # print(logits.shape)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # print('idx:', idx)
            # print('idx next:', idx_next)
            idx = torch.cat([idx, idx_next], dim=1)
            # gen_text.append(idx_next.item())
        return idx
    

x_batch, y_batch = get_batch('train', batch_size)
# print(x_batch.shape, y_batch.shape)
model = BigramLanguageModel(vocab_size, n_embd, n_blocks)
model = model.to(device)
logits, loss = model(x_batch, y_batch)
print(loss)

inp = torch.zeros((1,1), dtype=torch.long, device=device)
gen_text = model.generate(inp, max_new_tokens=500)[0]
print(decode_seq(gen_text.tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


def estimate_loss(eval_iters=100):
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for itr in range(eval_iters):
                x_batch, y_batch = get_batch(split, batch_size)
                _, loss = model(x_batch, y_batch)
                losses[itr] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

for itr in range(max_itr):
    if itr % eval_interval == 0:
        losses = estimate_loss(eval_iters)
        print(f"step: {itr}, train_loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    x_batch, y_batch = get_batch('train', batch_size)
    _, loss = model(x_batch, y_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# generate new text
inp = torch.zeros((1,1), dtype=torch.long, device=device)
gen_text = model.generate(inp, max_new_tokens=500)[0]
print(decode_seq(gen_text.tolist()))