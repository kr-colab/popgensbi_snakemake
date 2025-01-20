import numpy as np
import zarr
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence
from typing import Callable


class ZarrDataset(Dataset):
    def __init__(
        self, 
        zarr_path: str, 
        split: str, 
        use_cache: bool = True, 
        packed_sequence: bool = False,
    ):
        self.root = zarr.open(zarr_path, mode='r')
        self.use_cache = use_cache
        self.indices = self.root[split][:]
        self.x_data = self.root.features
        self.x_shape = self.root.features_shape
        self.theta = Tensor(np.stack(self.root.targets[self.indices]))

        # If x is ragged use a PackedSequence to collate into batches
        self.x_collate = torch.stack if not packed_sequence else \
            lambda x: pack_sequence(x, enforce_sorted=False) 
        # TODO: rather than torch.stack, use pad_sequence so that
        # ragged arrays are handled seemlessly

        # Preload data into CPU memory if requested
        if self.use_cache:
            # TODO: make this multithreaded
            self.x_cache = [
                torch.from_numpy(self.x_data[i].reshape(*self.x_shape[i])).float()
                for i in self.indices
            ]

    def __getitem__(self, idx: int) -> tuple:
        if self.use_cache:
            # Use preloaded tensors
            x = self.x_cache[idx]
        else:
            # Load from zarr and return CPU tensors
            i = self.indices[idx]
            x = torch.from_numpy(self.x_data[i].reshape(*self.x_shape[i])).float()
        return self.theta[idx], x

    def __len__(self) -> int:
        return len(self.indices)

    def make_collate_fn(self) -> Callable:
        return lambda data: tuple([
            torch.stack([d[0] for d in data]),
            self.x_collate([d[1] for d in data]),
        ])


class ArrayDataset(Dataset):
    def __init__(
        self, 
        features: np.ndarray,
        targets: np.ndarray,
    ):
        assert features.ndim == 2 and targets.ndim == 2
        assert features.shape[0] == targets.shape[0]
        self.features = features
        self.targets = targets

    def __getitem__(self, idx: int) -> tuple:
        return self.targets[idx], self.features[idx]

    def __len__(self) -> int:
        return self.features.shape[0]
