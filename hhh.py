from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import extra
from core.function import test_final
from dataset.dataset import get_dataset
from models.model import create_model
from utils.utils import create_logger
from utils.utils import get_tCAM


def calculate_if_fg(duration, segments):
    if_fg = np.zeros((config.DATASET.VIDEONUM_TEST, config.DATASET.NUM_SEGMENTS))

    for i in range(config.DATASET.VIDEONUM_TEST):
        for segment in segments[i]:
            if_fg[i, int(segment[0] / duration[i, 0] * config.DATASET.NUM_SEGMENTS) : int(segment[0] / duration[i, 0] * config.DATASET.NUM_SEGMENTS)] = 1

    return if_fg


def evaluate_correction(test_dataset_rgb, model_rgb, test_dataset_flow, model_flow, if_fg):
    with torch.no_grad():
        cnt_all = 0
        cnt = 0
        for i in range(config.DATASET.VIDEONUM_TEST):
            rgb_features = test_dataset_rgb.__getitem__(i, sampling_method='uniform')[0]
            rgb_features = torch.Tensor(rgb_features).cuda()

            # forward propagation
            rgb_attention = model_rgb.att_head(rgb_features).view(-1)
            rgb_class_w = model_rgb.clf_head.fc1.weight[0: config.DATASET.CLF_DIM - 1]

            # convert to numpy
            rgb_features = rgb_features.cpu().numpy()
            rgb_attention = rgb_attention.cpu().numpy()
            rgb_class_w = rgb_class_w.cpu().numpy().transpose()

            # get tCAM
            rgb_tCam = get_tCAM(rgb_features, rgb_class_w).max(axis=1)

            flow_features = test_dataset_flow.__getitem__(i, sampling_method='uniform')[0]
            flow_features = torch.Tensor(flow_features).cuda()

            # forward propagation
            flow_attention = model_flow.att_head(flow_features).view(-1)
            flow_class_w = model_flow.clf_head.fc1.weight[0: config.DATASET.CLF_DIM - 1]

            # convert to numpy
            flow_features = flow_features.cpu().numpy()
            flow_attention = flow_attention.cpu().numpy()
            flow_class_w = flow_class_w.cpu().numpy().transpose()

            # get tCAM
            flow_tCam = get_tCAM(flow_features, flow_class_w).max(axis=1)

            # threshold = 0.5
            attention = ((rgb_attention + flow_attention) / 2 > 0.5).astype(np.float)
            tCAM = ((rgb_tCam + flow_tCam) / 2 > 0.5).astype(np.float)

            # number of background frames with high clf score
            cnt_all += np.count_nonzero((tCAM > if_fg[i]).astype(np.float))

            # number of background frames with high attention
            cnt += np.count_nonzero((attention > if_fg[i]).astype(np.float))

    return cnt / cnt_all



def main():
    # convert to train mode
    config.MODE = 'test'
    extra()

    # create a logger
    logger = create_logger(config, 'test')

    # logging configurations
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # create a model
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpus = range(gpus.__len__())

    model_rgb = create_model()
    model_rgb.my_load_state_dict(torch.load(config.TEST.STATE_DICT_RGB), strict=True)
    model_rgb = model_rgb.cuda(gpus[0])
    # model_rgb = torch.nn.DataParallel(model_rgb, device_ids=gpus)

    model_flow = create_model()
    model_flow.my_load_state_dict(torch.load(config.TEST.STATE_DICT_FLOW), strict=True)
    model_flow = model_flow.cuda(gpus[0])
    # model_flow = torch.nn.DataParallel(model_flow, device_ids=gpus)

    # load data
    test_dataset_rgb = get_dataset(mode='test', modality='rgb')
    test_dataset_flow = get_dataset(mode='test', modality='flow')
    test_dataset_rgb.mode = 'train'
    test_dataset_flow.mode = 'train'

    test_loader_rgb = torch.utils.data.DataLoader(
        test_dataset_rgb,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )
    test_loader_flow = torch.utils.data.DataLoader(
        test_dataset_flow,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    duration = np.load(os.path.join(config.DATASET.DATAROOT,
                                    'Thumos14reduced-Annotations',
                                    'duration.npy'))[-config.DATASET.VIDEONUM_TEST:]
    segments = np.load(os.path.join(config.DATASET.DATAROOT,
                                    'Thumos14reduced-Annotations',
                                    'segments.npy'), allow_pickle=True)[-config.DATASET.VIDEONUM_TEST:]
    if_fg = calculate_if_fg(duration, segments)  # [videonum_test, num_segment]


    ratio = evaluate_correction(test_dataset_rgb, model_rgb, test_dataset_flow, model_flow, if_fg)

    logger.info(ratio)


if __name__ == '__main__':
    main()