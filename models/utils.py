from collections import deque
from functools import lru_cache
import math
import os
import random
import time
import warnings
import heapq
import numpy as np

# some of the utility methods are helpful for Torch
import torch
import torch.nn as nn
# default type in padded3d needs to be protected if torch
# isn't installed.
TORCH_LONG = torch.long
__TORCH_AVAILABLE = True


"""Near infinity, useful as a large penalty for scoring when inf is bad."""
NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

def _create_embeddings(dictionary, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    #e=nn.Embedding.from_pretrained(data, freeze=False, padding_idx=0).double()
    e = nn.Embedding(len(dictionary)+4, embedding_size, padding_idx)
    e.weight.data.copy_(torch.from_numpy(np.load('word2vec_redial.npy')))
    #nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    #e.weight=data
    #nn.init.constant_(e.weight[padding_idx], 0)
    return e


def _create_entity_embeddings(entity_num, embedding_size, padding_idx):
    """Create and initialize word embeddings."""
    e = nn.Embedding(entity_num, embedding_size)
    nn.init.normal_(e.weight, mean=0, std=embedding_size ** -0.5)
    nn.init.constant_(e.weight[padding_idx], 0)
    return e