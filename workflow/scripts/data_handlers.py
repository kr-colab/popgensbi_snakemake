import numpy as np
import zarr
import torch
import ray
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence
from typing import Callable, List
import os


class ZarrDataset(Dataset):
    """
    Dataset for loading data from Zarr format.

    When use_cache=True, uses Ray for parallel loading of data into CPU memory.
    """

    def __init__(
        self,
        zarr_path: str,
        split: str,
        use_cache: bool = False,  # Default to False based on benchmarks
        packed_sequence: bool = False,
        ray_batch_size: int = 100,  # Batch size for Ray parallel loading
        ray_num_workers: int = None,  # Number of Ray workers (None = auto)
    ):
        self.root = zarr.open(zarr_path, mode="r")
        self.use_cache = use_cache
        self.indices = self.root[split][:]
        self.x_data = self.root.features
        self.x_shape = self.root.features_shape
        self.zarr_path = zarr_path

        # Handle targets - support object arrays from zarr
        if "targets" in self.root:
            targets_data = self.root.targets[self.indices]
            if targets_data.dtype == object:
                # Need to stack object arrays
                self.theta = Tensor(np.stack(targets_data).astype(np.float32))
            else:
                self.theta = Tensor(targets_data.astype(np.float32))
        else:
            self.theta = None

        # If x is ragged use a PackedSequence to collate into batches
        self.x_collate = (
            torch.stack
            if not packed_sequence
            else lambda x: pack_sequence(x, enforce_sorted=False)
        )

        # Preload data into CPU memory if requested -  using Ray
        if self.use_cache:
            if ray_num_workers is None:
                ray_num_workers = min(os.cpu_count() or 4, 8)

            # Initialize Ray if not already initialized
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, num_cpus=ray_num_workers)

            # Load cache in parallel
            self.x_cache = self._parallel_load_cache(ray_num_workers, ray_batch_size)

    def _parallel_load_cache(
        self, num_workers: int, batch_size: int
    ) -> List[torch.Tensor]:
        """Load cache using Ray parallel processing."""

        # Create batches of indices
        n_samples = len(self.indices)
        batches = []
        for i in range(0, n_samples, batch_size):
            batch_indices = self.indices[i : i + batch_size]
            batch_positions = list(range(i, min(i + batch_size, n_samples)))
            batches.append((batch_indices, batch_positions))

        # Define Ray remote function
        @ray.remote
        def load_batch(
            zarr_path: str, indices: np.ndarray, positions: List[int]
        ) -> List[tuple]:
            """Load a batch of samples from zarr."""
            # Open zarr in this worker
            root = zarr.open(zarr_path, mode="r")
            x_data = root.features
            x_shape = root.features_shape

            results = []
            for idx, pos in zip(indices, positions):
                # Load and convert data
                arr = x_data[idx].reshape(*x_shape[idx])
                # Use torch.tensor to create a copy and avoid non-writable warning
                tensor = torch.tensor(arr, dtype=torch.float32)
                results.append((pos, tensor))

            return results

        # Submit all batches to Ray
        futures = []
        for batch_indices, batch_positions in batches:
            future = load_batch.remote(self.zarr_path, batch_indices, batch_positions)
            futures.append(future)

        # Collect results
        cache = [None] * n_samples
        for future in ray.get(futures):
            for pos, tensor in future:
                cache[pos] = tensor

        return cache

    def __getitem__(self, idx: int) -> tuple:
        if self.use_cache:
            # Use preloaded tensors
            x = self.x_cache[idx]
        else:
            # Load from zarr and return CPU tensors
            i = self.indices[idx]
            arr = self.x_data[i].reshape(*self.x_shape[i])
            # Use torch.tensor to avoid non-writable numpy array warning
            x = torch.tensor(arr, dtype=torch.float32)
        if self.theta is not None:
            return self.theta[idx], x
        else:
            # return empty tensor for theta
            return torch.empty(()), x

    def __len__(self) -> int:
        return len(self.indices)

    def make_collate_fn(self) -> Callable:
        return lambda data: tuple(
            [
                torch.stack([d[0] for d in data]),
                self.x_collate([d[1] for d in data]),
            ]
        )


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
