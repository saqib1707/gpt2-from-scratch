import os
import math
import numpy as np
import time
from dataclasses import dataclass
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# import code; code.interact(local=locals())

from model import GPT
from dataloader import DataLoaderLite
from hellaswag import render_example, iterate_examples, get_most_likely_row

torch.set_float32_matmul_precision('high')    # enable TF32 precision

# torch compile results in error for me
use_torch_compile = False


class Trainer:
    def __init__(
            self, 
            model, 
            optimizer, 
            train_loader, 
            val_loader, 
            token_encoder, 
            eval_freq, 
            grad_accum_steps, 
            ddp, 
            ddp_rank, 
            ddp_world_size, 
            device, 
            logpath
    ):
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.master_process = ddp_rank == 0
        self.ddp_world_size = ddp_world_size

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.token_encoder = token_encoder

        self.eval_freq = eval_freq
        self.grad_accum_steps = grad_accum_steps
        self.device = device
        self.device_type = 'cuda' if device.startswith('cuda') else 'cpu'
        self.logpath = logpath


    def train(
        self, 
        max_steps, 
        warmup_steps, 
        max_lr, 
        min_lr
    ):
        for step in range(max_steps):
            t0 = time.time()
            self.is_last_step = (step == max_steps - 1)

            # evaluate validation loss
            if step % self.eval_freq == 0 or self.is_last_step:
                self.evaluate_validation(step)

            # evaluate model performance on HellaSwag every once in a while
            if ((step > 0 and step % self.eval_freq == 0) or self.is_last_step) and (not use_torch_compile):
                self.evaluate_helloswag(step)

            # generate sequences from the model every once in a while
            if ((step > 0 and step % self.eval_freq == 0) or self.is_last_step) and (not use_torch_compile):
                self.generate_sequences(num_seq=5, max_tokens=32)

            # training loop starts here
            self.model.train()    # sets model to train mode
            self.optimizer.zero_grad()    # resets all gradients
            batch_loss = 0.0
            
            for mini_step in range(self.grad_accum_steps):
                inp, tar = self.train_loader.next_batch()
                inp, tar = inp.to(self.device), tar.to(self.device)
                
                # FORWARD PASS !!!
                # autocast to bfloat16 for faster compute and memory efficiency
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(inp, tar)

                # loss is scaled to account for gradient accumulation, because the gradients just add
                # on each successive backward() call. Addition of gradients corresponds to SUM in the objective,
                # but we want MEAN instead of a SUM
                loss /= self.grad_accum_steps
                batch_loss += loss.detach()

                if self.ddp:
                    # in the final mini_step, sync and avg all gradients across all processes. used by both forward and backward processes
                    # can use 'no_sync()' context manager alternatively. 
                    self.model.require_backward_grad_sync = (mini_step == self.grad_accum_steps - 1)

                # each process accumulates gradients separately when 'require_backward_grad_sync'=False
                # in the final 'mini_step', 'require_backward_grad_sync' becomes True, therefore 
                # gradients are averaged across all processes and shared among them by loss.backward()
                loss.backward()

            if self.ddp:
                # 'batch_loss' is outside of DDP container, so need to perform 'all_reduce' to 
                # average out 'batch_loss' across all processes of all ranks. 'batch_loss' tensor exists on all GPUs. 
                # 'all_reduce' averages and deposits the result on all the processes
                dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)

            # once gradients are computed, clip the global l2-norm of the gradient at 1.0
            norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)    # monitor/print 'norm'

            # determine learning rate with decay
            lr = self.estimate_lr(step, warmup_steps, max_steps, max_lr, min_lr)
            # set learning rate for this iteration
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            self.optimizer.step()
            if self.device_type == 'cuda':
                torch.cuda.synchronize()    # wait for the GPU to finish work
            
            dt = (time.time() - t0) * 1000.0    # in ms
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_world_size
            tokens_per_sec = tokens_processed / dt

            if self.master_process:
                print(f'step {step:4d} | loss: {batch_loss.item():.6f} | lr: {lr:.2e} | norm: {norm:.4f} | dt: {dt:.4f}ms | tok/sec: {tokens_per_sec:.4f}')
                with open(self.logpath, 'a') as f:
                    f.write(f'{step} train {batch_loss.item():.6f}\n')


    def evaluate_validation(self, step):
        self.model.eval()    # sets model to eval mode
        self.val_loader.reset()
        # evaluate the model on validation set
        with torch.no_grad():
            val_loss_accum = 0.0
            val_steps = 20
            for _ in range(val_steps):
                inp, tar = self.val_loader.next_batch()
                inp, tar = inp.to(self.device), tar.to(self.device)
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(inp, tar)
                loss /= val_steps
                val_loss_accum += loss.detach()

        if self.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.master_process:
            print(f'Val loss: {val_loss_accum.item():.4f}')
            with open(self.logpath, 'a') as f:
                f.write(f'{step} val {val_loss_accum.item():.4f}\n')

            if step > 0 and (step % 10000 == 0 or self.is_last_step):
                raw_model = self.model.module if self.ddp else self.model
                logdir = os.path.dirname(self.logpath)
                ckpt_path = os.path.join(logdir, f'model_{step:05d}.pt')
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }    # add optimizer.state_dict(), rng_seeds, etc. if resuming training
                torch.save(checkpoint, ckpt_path)


    def evaluate_helloswag(self, step):
        """ 
        Construct a batch of 4 sequences and perform token completion using 
        our model. 
        """
        n_total = 0
        n_correct_norm = 0
        for i, example in enumerate(iterate_examples('val')):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % self.ddp_world_size != self.ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)    # (4,N), (4,N), (4,N)
            tokens, mask = tokens.to(self.device), mask.to(self.device)
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            n_total += 1
            n_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if self.ddp:
            n_total = torch.tensor(n_total, device=self.device, dtype=torch.long)
            n_correct_norm = torch.tensor(n_correct_norm, device=self.device, dtype=torch.long)
            dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_correct_norm, op=dist.ReduceOp.SUM)
            n_total = n_total.item()
            n_correct_norm = n_correct_norm.item()
        acc_norm = n_correct_norm / n_total
        if self.master_process:
            print(f'HelloSwag accuracy: {n_correct_norm}/{n_total}={acc_norm:.4f}')
            with open(self.logpath, 'a') as f:
                f.write(f'{step} hellaswag {acc_norm:.4f}\n')


    def generate_sequences(self, num_seq=4, max_tokens=32):
        self.model.eval()
        tokens = self.token_encoder.encode("Hello, I am a language model")
        tokens = torch.tensor(tokens, dtype=torch.long)    # (n,)   n : current sequence length
        tokens = tokens.unsqueeze(0).repeat(num_seq, 1)    # (1,n) --> (num_seq, n)
        gen_tokens = tokens.to(self.device)
        # create a different rng generator so as not to impact the global rng state used for training
        sample_rng = torch.Generator(device=self.device)
        # adding 'ddp_rank' in seeding to generate different tokens for different rank processes
        sample_rng.manual_seed(42 + self.ddp_rank)
        # generate new tokens one token at a time until the sequence length becomes 'max_tokens'
        while gen_tokens.shape[-1] <= max_tokens:
            with torch.no_grad():
                with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(gen_tokens)    # (num_seq, n, vocab_size)
                logits = logits[:, -1, :]    # (num_seq, vocab_size)
                probs = F.softmax(logits, dim=-1)    # (num_seq, vocab_size)
                # take top-k 50 probs
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # (num_seq, 50), (num_seq, 50)
                # sample a token from top-50 probabilities
                ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)    # (num_seq, 1)
                next_tok = torch.gather(topk_indices, -1, ix)    # (num_seq, 1)
                gen_tokens = torch.cat([gen_tokens, next_tok], dim=1)
        # decode generated tokens and print generated text
        for i in range(num_seq):
            tokens = gen_tokens[i, :max_tokens].tolist()
            gen_text = self.token_encoder.decode(tokens)
            print(f"> rank {self.ddp_rank} sample {i}: {gen_text}")


    def estimate_lr(self, step, warmup_steps, max_steps, max_lr, min_lr):
        """
        Learning rate scheduler: Cosine-decay learning schedule with warmup
        """
        # 1) linear warmup for 'warmup_iters' steps
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


@dataclass
class GPTConfig:
    context_length: int = 1024    # max context / sequence length
    vocab_size: int = 50257    # number of tokens: 50000 BPE merges + 256 bytes tokens + 1 <endoftext> token
    num_layers: int = 12
    embd_size: int = 768    # embedding dim
    num_heads: int = 12


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter Configuration")
    parser.add_argument("--total_batch_size", type=int, default=524288, help="number of tokens processed for each weight update")    # =2^19 tokens/step update, (~0.5M tokens used in openai gpt3 paper)
    parser.add_argument("--mini_batch_size", type=int, default=64, help="setting of mini_batch_size is just a performance optimization. bigger gpu, bigger mini_batch_size")
    parser.add_argument("--context_length", type=int, default=1024)    # max sequence length (can also try 2048)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embd_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-3 * 0.1)
    parser.add_argument("--warmup_steps", type=int, default=715)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=19073)    # 10^10 / 2^19 ~ 19073 for 1 epoch on FineWebEdu-sample10BT
    parser.add_argument("--eval_freq", type=int, default=250)
    # parser.add_argument("--use_torch_compile", action='store_true')    # default False
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")
    parser.add_argument("--logdir", type=str, default="./logs_124M_50B/")
    return parser.parse_args()


def main():
    args = get_args()

    # Print the hyperparameters
    print("Hyperparameter Configuration:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    # create the logs directory if it doesn't exist
    os.makedirs(args.logdir, exist_ok=True)
    logpath = os.path.join(args.logdir, 'log.txt')
    with open(logpath, 'w') as f:
        pass

    # set up DDP (distributed data parallel)
    # 'torchrun' command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    # RANK and LOCAL_RANK same for (single node, multi-GPU) settings, may differ for (multinode, 
    # multi GPU) settings. 
    ddp = int(os.environ.get('RANK', -1)) != -1    # if this is a ddp run or not
    if ddp:
        # use of ddp requires CUDA
        assert torch.cuda.is_available(), f'use of DDP requires CUDA'
        dist.init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        # master process (arbitrarily set to 0) will do printing, logging, checkpointing, etc.
        master_process = ddp_rank == 0
    else:
        # not using ddp
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True    # ddp_rank == 0
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'    # for apple macbook GPUs
        print(f'using device: {device}')

    device_type = 'cuda' if device.startswith('cuda') else 'cpu'

    # setting seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    # sets seed for random number generation on CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)    # sets seed for random number generation on GPU
        torch.cuda.manual_seed_all(args.seed)    # sets seed for all GPUs

    assert args.total_batch_size % (args.mini_batch_size * args.context_length * ddp_world_size) == 0, f'ensure total_batch_size divisible by B*T*ddp_world_size'
    grad_accum_steps = args.total_batch_size // (args.mini_batch_size * args.context_length * ddp_world_size)
    if master_process:
        print(f'desired batch size (number of tokens): {args.total_batch_size}')
        print(f'gradient accumulation steps: {grad_accum_steps}')
    print(f'GPU: {ddp_rank}, {ddp_local_rank}')

    train_loader = DataLoaderLite(B=args.mini_batch_size, T=args.context_length, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
    val_loader = DataLoaderLite(B=args.mini_batch_size, T=args.context_length, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

    # create GPT model. each ddp process will create its own instance of the model but since the seed is fixed, 
    # they will create same identical model
    gpt_config = GPTConfig(vocab_size=50304,   # 50304 (nice number, lots of power of 2s) used instead of 50257 (bad, odd number)
                           context_length=args.context_length, 
                           num_layers=args.num_layers, 
                           num_heads=args.num_heads, 
                           embd_size=args.embd_size
                           )
    model = GPT(config=gpt_config)
    # model = GPT.from_pretrained('gpt2')    # init from OpenAI GPT-2
    model.to(device)    # move model to device
    if use_torch_compile:
        # use torch compile almost always unless debugging (requires compilation time, but makes training faster)
        # speedup comes from reducing python overhead and GPU read/write
        model = torch.compile(model)

    if ddp:
        # wraps the model in DDP container (forward pass is unchanged, but after backward pass,
        # gradients computed across each processes averaged by DDP using 'AllReduce' and shared across 
        # all processes so that each process has same gradients)
        model = DDP(model, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay, lr=args.max_lr, device_type=device_type, master_process=master_process)
    token_encoder = tiktoken.get_encoding('gpt2')

    start_time = time.time()
    # init the trainer object
    trainer = Trainer(model, optimizer, train_loader, val_loader, token_encoder, args.eval_freq, grad_accum_steps, 
                      ddp, ddp_rank, ddp_world_size, device, logpath)

    max_steps = args.steps_per_epoch * args.num_epochs
    trainer.train(max_steps, args.warmup_steps, args.max_lr, args.min_lr)

    dt = (time.time() - start_time) / (60*60)
    print(f"Total training time: {dt:.4f}hr")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
