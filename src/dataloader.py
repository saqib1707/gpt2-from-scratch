import os
import numpy as np
import torch

script_dir = os.path.dirname(__file__)


class DataLoaderLite:
    """ A simple dataloader for FineWebEdu-10B dataset """

    def __init__(self, B, T, process_rank, num_processes, split='train'):
        super().__init__()
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
        master_process = process_rank == 0
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