from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

import torch
import torch.nn as nn
import torch.nn.functional as F

import _init_paths
from core.config import config
from utils.utils import get_tCAM

logger = logging.getLogger(__name__)


def loss_fg(x, y):
    """
    foreground classification loss (y could be multi-label)
    """
    gt_prob = y / y.sum(dim=1, keepdim=True)
    return -(gt_prob * F.log_softmax(x, dim=1)).sum(dim=1).mean()


# def loss_bg(x, y=torch.ones((1, config.DATASET.CLF_DIM)).cuda()):
#     """
#     background classification loss (y is uniform if not specified)
#     """
#     gt_prob = y / y.sum(dim=1, keepdim=True)
#     return -(gt_prob * F.log_softmax(x, dim=1)).sum(dim=1).mean()


def loss_bg(x, y=torch.cat([torch.zeros((1, config.DATASET.CLF_DIM - 1)), torch.ones((1, 1))], dim=1).cuda()):
    """
    background classification loss (y is uniform if not specified)
    """
    gt_prob = y / y.sum(dim=1, keepdim=True)
    return -(gt_prob * F.log_softmax(x, dim=1)).sum(dim=1).mean()


def loss_sparse(attention):
    """
    sparsity loss
    """
    return attention.norm(p=1, dim=1).mean()


def loss_guide(x, y, attention, clf_weight):
    """
    self-guide loss
    """
    # tCAM = [batch_size, segment_num, clf_dim]
    # x = [batch_size, segment_num, feature_dim]
    tCAM = (x.view(-1, config.DATASET.FEATURE_DIM) @ clf_weight).view(-1, config.DATASET.NUM_SEGMENTS,
                                                                      config.DATASET.CLF_DIM)
    tCAM = torch.softmax(tCAM, dim=2)
    # tCAM = torch.sigmoid(tCAM)
    tCAM = tCAM.detach().cpu().numpy()
    tCAM = gaussian_filter1d(tCAM, sigma=1, axis=1)
    tCAM = torch.Tensor(tCAM).cuda()

    # foreground guidance
    guidance_fg = (tCAM * y.view(-1, 1, config.DATASET.CLF_DIM)).max(dim=2)[0]

    # background guidance
    guidance_bg = 1 - tCAM[:, :, -1]

    loss_guide = ((attention - guidance_fg).abs() + (attention - guidance_bg).abs()).mean()
    # loss_guide = (attention - guidance_fg).abs().mean()
    return loss_guide


def loss_cluster(gap_fg, gap_bg):
    """
    clustering loss
    :param gap_fg: (u_fg - u_bg) * x_fg
    :param gap_bg: (u_bg - u_fg) * x_bg
    """
    y = torch.ones_like(gap_fg).cuda()
    loss_cluster_fg = F.binary_cross_entropy_with_logits(gap_fg, y)
    loss_cluster_bg = F.binary_cross_entropy_with_logits(gap_bg, y)
    return loss_cluster_fg + loss_cluster_bg


def loss_ising(attention):
    attention = attention.view(-1, config.DATASET.NUM_SEGMENTS)
    return (attention[:, 1:] - attention[:, :-1]).pow(2).mean()


def loss_cvae(recon_x, x, mean, log_var, attention):
    """
    loss of conditional-VAE
    :param recon_x: reconstructed x
    :param x: original x
    :param mean: mean of Gaussian distribution of latent variable z
    :param log_var: log(variance) of Gaussian distribution of latent variable z
    :return:
    """
    recon_x = recon_x.view(-1, config.DATASET.FEATURE_DIM)
    x = x.view(-1, config.DATASET.FEATURE_DIM)
    mean = (mean - attention).view(-1, mean.shape[-1])
    log_var = log_var.view(-1, log_var.shape[-1])

    mse = (x - recon_x).pow(2).mean()
    kld = -0.5 * (1 + log_var - mean.pow(2) - log_var.exp()).mean()

    logger.info(mse)
    logger.info(kld)

    return 0.1 * kld + 1 * mse