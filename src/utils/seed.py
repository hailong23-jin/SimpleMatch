import numpy as np
import random
import torch
import time

def init_random_seed(seed=None, device='cuda'):
    if seed is not None:
        return seed
    seed = int(random.random() * 1e5)
    return seed


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True