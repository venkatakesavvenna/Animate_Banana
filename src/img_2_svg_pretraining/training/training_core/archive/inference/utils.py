import torch
import torch.multiprocessing as mp

class ShardedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, shard_id, num_shards):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))[shard_id::num_shards]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]