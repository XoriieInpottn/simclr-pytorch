#!/usr/bin/env python3


"""
@author: xi
"""

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):

    def __init__(self, emb_size, head_size):
        super(ProjectionHead, self).__init__()
        self.hidden = nn.Linear(emb_size, emb_size)
        self.out = nn.Linear(emb_size, head_size)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.hidden(h)
        h = F.relu_(h)
        h = self.out(h)
        return h


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, t=0.3, eps=1e-10) -> torch.Tensor:
    batch_size = z1.shape[0]
    assert batch_size == z2.shape[0]
    assert batch_size > 1

    # compute the similarity matrix
    # values in the diagonal elements represent the similarity between the (POS, POS) pairs
    # while the other values are the similarity between the (POS, NEG) pairs
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim_mat = z1 @ z2.T
    scaled_prob_mat = F.softmax(sim_mat / t, dim=1)

    # construct a cross-entropy loss to maximize the probability of the (POS, POS) pairs
    log_prob = torch.log(scaled_prob_mat + eps)
    return -torch.diagonal(log_prob).mean()
