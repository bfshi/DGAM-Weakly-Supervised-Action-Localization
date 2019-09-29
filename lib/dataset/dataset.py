from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import csv
import cv2
import os
import glob

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from utils.utils import random_perturb
from utils.utils import uniform_sampling

NUM_SEGMENTS = config.DATASET.NUM_SEGMENTS

class THUMOS14(Dataset):
    def __init__(self, mode, modality):
        self.video_num = config.DATASET.VIDEONUM_TRAIN if mode == 'train' else config.DATASET.VIDEONUM_TEST
        self.mode = mode
        self.modality = modality
        self.feature_list = []
        self.labels = None

        self.load_dataset()

    def load_dataset(self):
        for i in range(self.video_num):
            feature_path = os.path.join(config.DATASET.DATAROOT,
                                        '{}_data'.format(self.mode),
                                        '{}_features'.format(self.modality),
                                        '{:d}.npy'.format(i + 1))
            self.feature_list.append(np.load(feature_path))

        labels_path = os.path.join(config.DATASET.DATAROOT,
                                   '{}_data'.format(self.mode),
                                   '{}_labels.npy'.format(self.mode))
        self.labels = np.load(labels_path)

    def __len__(self):
        return self.video_num

    def __getitem__(self, idx, sampling_method = 'random'):
        """
        :param idx: index of the video
        :return: feature, label(binary vector)
        """
        feature = self.feature_list[idx]
        label = np.concatenate([self.labels[idx], np.zeros(1)])
        if self.mode == 'train':# or config.MODE == 'train':
            config.DATASET.NUM_SEGMENTS = NUM_SEGMENTS = 400
        else:
            config.DATASET.NUM_SEGMENTS = NUM_SEGMENTS = max(min(feature.shape[0], 400), 50)

            # randomly sample a fixed number of segments
        if sampling_method == 'random':
            selected_seg_list = random_perturb(feature.shape[0], NUM_SEGMENTS)
        elif sampling_method == 'uniform':
            selected_seg_list = uniform_sampling(feature.shape[0], NUM_SEGMENTS)
        sampled_feature = feature[selected_seg_list]  # [NUM_SEGMENTS, FEATURE_DIM]

        return sampled_feature, label

    def get_video_len(self, idx):
        return self.feature_list[idx].shape[0]


class ActivityNet12(Dataset):
    def __init__(self, mode, modality):
        self.video_num = config.DATASET.VIDEONUM_TRAIN if mode == 'train' else config.DATASET.VIDEONUM_TEST
        self.mode = mode
        self.modality = modality
        self.feature_list = []
        self.labels = None

        self.load_dataset()

    def load_dataset(self):
        for i in range(self.video_num):
            feature_path = os.path.join(config.DATASET.DATAROOT,
                                        '{}_data'.format(self.mode),
                                        '{}_features'.format(self.modality),
                                        '{:d}.npy'.format(i + 1))
            self.feature_list.append(np.load(feature_path))

        labels_path = os.path.join(config.DATASET.DATAROOT,
                                   '{}_data'.format(self.mode),
                                   '{}_labels.npy'.format(self.mode))
        self.labels = np.load(labels_path)

    def __len__(self):
        return self.video_num

    def __getitem__(self, idx, sampling_method = 'random'):
        """
        :param idx: index of the video
        :return: feature, label(binary vector)
        """
        feature = self.feature_list[idx]
        label = np.concatenate([self.labels[idx], np.zeros(1)])

        if self.mode == 'train':
            config.DATASET.NUM_SEGMENTS = NUM_SEGMENTS = 400
        elif self.mode == 'test':
            config.DATASET.NUM_SEGMENTS = NUM_SEGMENTS = min(max(feature.shape[0], 2), 200)

        # randomly sample a fixed number of segments
        if sampling_method == 'random':
            selected_seg_list = random_perturb(feature.shape[0], NUM_SEGMENTS)
        elif sampling_method == 'uniform':
            selected_seg_list = uniform_sampling(feature.shape[0], NUM_SEGMENTS)
        sampled_feature = feature[selected_seg_list]  # [NUM_SEGMENTS, FEATURE_DIM]

        return sampled_feature, label

    def get_video_len(self, idx):
        return self.feature_list[idx].shape[0]


def get_dataset(mode, modality):
    """
    :param mode: 'train' / 'test'
    :param modality: 'rgb' / 'flow'
    :return: dataset
    """
    dataset = eval(config.DATASET_NAME)(mode, modality)
    return dataset