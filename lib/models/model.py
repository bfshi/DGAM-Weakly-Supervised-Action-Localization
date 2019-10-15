from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from collections import OrderedDict

import _init_paths
from core.config import config


ATT_IM_DIM = 256  # intermediate output dimension of Att_Head


class Clf_Head(nn.Module):
    def __init__(self):
        super(Clf_Head, self).__init__()

        self.fc1 = nn.Linear(config.DATASET.FEATURE_DIM, config.DATASET.CLF_DIM)

    def forward(self, x):
        return self.fc1(x)


class Att_Head(nn.Module):
    def __init__(self):
        super(Att_Head, self).__init__()

        self.fc1 = nn.Linear(config.DATASET.FEATURE_DIM, ATT_IM_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ATT_IM_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Cluster_Head(nn.Module):
    def __init__(self):
        super(Cluster_Head, self).__init__()

        self.u_fg = nn.Linear(config.DATASET.FEATURE_DIM, 1)
        self.u_bg = nn.Linear(config.DATASET.FEATURE_DIM, 1)

    def forward(self, feature_fg, feature_bg):
        return [self.u_fg(feature_fg) - self.u_bg(feature_fg), self.u_bg(feature_bg) - self.u_fg(feature_bg)]



class WSAL_Model(nn.Module):
    def __init__(self):
        super(WSAL_Model, self).__init__()

        self.clf_head = Clf_Head()
        self.att_head = Att_Head()
        self.cluster_head = Cluster_Head()

    def my_load_state_dict(self, state_dict_old, strict=True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)

    def forward(self, x, mode):
        """
        mode: 'clf' / 'att' / 'cluster'
        """
        if mode == 'clf':
            return self.clf_head(x)
        elif mode == 'att':
            return self.att_head(x)
        elif mode == 'cluster':
            # x: [feature_fg, feature_bg]
            return self.cluster_head(x[0], x[1])


################################   Conditional VAE   #################################


latent_size = 128
# condition_len = 1 + (config.DATASET.FEATURE_DIM + 1) * config.MODEL.CONDITION_FRAME_NUM

class CEncoder(nn.Module):
    def __init__(self):
        super(CEncoder, self).__init__()

        self.fc1 = nn.Linear(config.DATASET.FEATURE_DIM + 1, latent_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(latent_size, latent_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(latent_size, 2 * latent_size)  # mean + log_var

    def forward(self, x, att):
        att = att.view(-1, config.DATASET.NUM_SEGMENTS, 1)  # [batch_size, seg_num, 1]
        x = torch.cat([x, att], dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x[:, :, :latent_size], x[:, :, latent_size:]


class CDecoder(nn.Module):
    def __init__(self):
        super(CDecoder, self).__init__()

        self.fc1 = nn.Linear(latent_size + 1, latent_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(latent_size, latent_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(latent_size, config.DATASET.FEATURE_DIM)

    def forward(self, z, att):
        att = att.view(-1, config.DATASET.NUM_SEGMENTS, 1)  # [batch_size, seg_num, 1]
        x = torch.cat([z, att], dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x



class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()

        self.cencoder = CEncoder()
        self.cdecoder = CDecoder()

    def forward(self, mode, x=None, att=None):
        if mode == 'forward':
            means, log_var = self.cencoder(x, att)

            std = torch.exp(0.5 * log_var)
            eps = torch.randn(means.shape, device='cuda')
            # eps = torch.randn(means.shape).cuda()
            z = means + eps * std

            recon_x = self.cdecoder(z, att)

            return means, log_var, z, recon_x
        elif mode == 'inference':
            att = att.view(-1, config.DATASET.NUM_SEGMENTS)
            z = torch.randn((*att.shape, latent_size), device='cuda') + att.view(-1, config.DATASET.NUM_SEGMENTS, 1)
            # z = torch.randn((*att.shape, latent_size)).cuda()
            recon_x = self.cdecoder(z, att)

            return recon_x



def create_model():
    model = WSAL_Model()
    return model

def create_cvae():
    cvae = CVAE()
    return cvae
