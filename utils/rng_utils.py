import numpy as np
import torch as th


def set_global_seed(seed):
    np.random.seed(seed)
    th.manual_seed(seed)