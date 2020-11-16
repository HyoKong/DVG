import numpy as np

import torch
import torch.nn as nn


def klLoss(mu, logvar, priorMu=0):
    vKl = mu.add(-priorMu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    vKl = vKl.sum(dim=-1).mul_(-0.5)  # [b, 2]
    return vKl


def reconLoss(prediction, target, sizeAvg=False):
    err = (prediction - target).view(prediction.size(0), -1)
    err = torch.abs(err)
    # err = err ** 2
    err = torch.sum(err, dim=-1)

    if sizeAvg:
        err = err.mean()
    else:
        err = err.sum()
    return err