
from typing import Dict
import torch


def to_cuda(batch: Dict):
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            batch[key] = val.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()