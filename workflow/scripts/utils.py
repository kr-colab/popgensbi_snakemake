"""
Utility functions for the workflow.
"""
import torch

def get_least_busy_gpu():
    """
    Get the least busy GPU.
    """
    if not torch.cuda.is_available():
        return None
    free_memory = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return free_memory.index(max(free_memory))