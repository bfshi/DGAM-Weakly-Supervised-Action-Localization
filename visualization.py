from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import logging
import time
import os
import json
import random

import numpy as np
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import _init_paths
from core.config import config
from core.config import extra
from dataset.dataset import get_dataset
from models.model import create_model
from utils.utils import create_logger
from utils.utils import get_tCAM, get_wtCAM, upgrade_resolution, interpolated_wtCAM, get_tempseg_list, get_temp_proposal
from utils.utils import integrated_prop, result2json, json2txt, inf_progress
from utils.utils import bar_figure

NUM_SEGMENTS = config.DATASET.NUM_SEGMENTS
SAMPLING_FRAMES = config.DATASET.SAMPLING_FRAMES
NUM_INPUT_FRAMES = config.DATASET.NUM_INPUT_FRAMES
scale = config.TEST.SCALE

sns.set(font_scale=2.5, rc={"lines.linewidth": 5, "legend.fontsize": 36})
sns.set_style("dark")


def test(test_dataset_rgb, model_rgb, test_dataset_flow, model_flow):
    # some hyper-parameters
    scale = config.TEST.SCALE
    class_threshold = config.TEST.CLASS_THRESHOLD

    # file for matching 'video number' and 'video name'
    test_vid_list = open(os.path.join(config.DATASET.DATAROOT, 'test_data',
                                      config.DATASET.TEST_VID_LIST), 'r')
    lines = test_vid_list.read().splitlines()

    # ground truth
    segments = np.load(os.path.join(config.DATASET.DATAROOT,
                                    'Thumos14reduced-Annotations',
                                    'segments.npy'), allow_pickle=True)[-(config.DATASET.VIDEONUM_TEST + 2):]
    segments = np.delete(segments, [26, 202])

    # Define json File (output)
    final_result = []

    with torch.no_grad():
        for i in range(config.DATASET.VIDEONUM_TEST):
            vid_name = lines[i]

            # labels: whether to test proposals of corresponding modality
            r_check = False
            f_check = False

            # if both modality are available
            if test_dataset_rgb is not None and test_dataset_flow is not None:
                rgb_features = test_dataset_rgb.__getitem__(i, sampling_method='uniform')[0]
                rgb_features = torch.Tensor(rgb_features).cuda()
                vid_len = test_dataset_rgb.get_video_len(i)

                # forward propagation
                rgb_attention = model_rgb.att_head(rgb_features)
                # rgb_features_fg = (rgb_features * rgb_attention).mean(dim=0, keepdim=True)
                rgb_features_fg = (rgb_features * rgb_attention).sum(dim=0, keepdim=True) / rgb_attention.sum(dim=0, keepdim=True)
                rgb_class_result = F.softmax(model_rgb.clf_head(rgb_features_fg))[:, 0: config.DATASET.CLF_DIM - 1]
                rgb_class_w = model_rgb.clf_head.fc1.weight[0: config.DATASET.CLF_DIM - 1]

                # convert to numpy
                rgb_features = rgb_features.cpu().numpy()
                rgb_attention = rgb_attention.cpu().numpy()
                rgb_class_result = rgb_class_result.cpu().numpy()
                rgb_class_w = rgb_class_w.cpu().numpy().transpose()

                # Gathering Classification Result
                if rgb_class_result.max() > 0.1:
                    rgb_class_prediction = np.where(rgb_class_result > class_threshold)[1]
                else:
                    rgb_class_prediction = np.array(np.argmax(rgb_class_result), dtype=np.int).reshape(-1)

                # get tCAM
                rgb_tCam = get_tCAM(rgb_features, rgb_class_w)

                flow_features = test_dataset_flow.__getitem__(i, sampling_method='uniform')[0]
                flow_features = torch.Tensor(flow_features).cuda()
                vid_len = test_dataset_flow.get_video_len(i)

                # forward propagation
                flow_attention = model_flow.att_head(flow_features)
                # flow_features_fg = (flow_features * flow_attention).mean(dim=0, keepdim=True)
                flow_features_fg = (flow_features * flow_attention).sum(dim=0, keepdim=True) / flow_attention.sum(dim=0, keepdim=True)
                flow_class_result = F.softmax(model_flow.clf_head(flow_features_fg))[:, 0: config.DATASET.CLF_DIM - 1]
                flow_class_w = model_flow.clf_head.fc1.weight[0: config.DATASET.CLF_DIM - 1]

                # convert to numpy
                flow_features = flow_features.cpu().numpy()
                flow_attention = flow_attention.cpu().numpy()
                flow_class_result = flow_class_result.cpu().numpy()
                flow_class_w = flow_class_w.cpu().numpy().transpose()

                # Gathering Classification Result
                if flow_class_result.max() > 0.1:
                    flow_class_prediction = np.where(flow_class_result > class_threshold)[1]
                else:
                    flow_class_prediction = np.array(np.argmax(flow_class_result), dtype=np.int).reshape(-1)

                # get tCAM
                flow_tCam = get_tCAM(flow_features, flow_class_w)

                # generate proposals
                if rgb_class_prediction.any():
                    r_check = True
                    # Weighted T-CAM
                    rgb_wtCam = get_wtCAM(rgb_tCam, rgb_attention, rgb_class_prediction, flow_tCam, config.TEST.FRAME_SCORE_RATIO_RGB)
                    # Interpolate W-TCAM
                    rgb_int_wtCam = interpolated_wtCAM(rgb_wtCam, scale)
                    rgb_int_attention = upgrade_resolution(rgb_attention, scale)
                    # Get segment list of rgb_int_wtCam
                    rgb_temp_idx = get_tempseg_list(rgb_int_wtCam, len(rgb_class_prediction), rgb_int_attention, thr=config.TEST.TEMPSEG_LIST_THR_RGB)
                    # Temporal Proposal
                    rgb_temp_prop = get_temp_proposal(rgb_temp_idx, rgb_int_wtCam, rgb_int_attention, rgb_class_prediction,
                                                       scale, vid_len)

                # generate proposals
                if flow_class_prediction.any():
                    f_check = True
                    # Weighted T-CAM
                    flow_wtCam = get_wtCAM(flow_tCam, flow_attention, flow_class_prediction, rgb_tCam, config.TEST.FRAME_SCORE_RATIO_FLOW)
                    # Interpolate W-TCAM
                    flow_int_wtCam = interpolated_wtCAM(flow_wtCam, scale)
                    flow_int_attention = upgrade_resolution(flow_attention, scale)
                    # Get segment list of flow_int_wtCam
                    flow_temp_idx = get_tempseg_list(flow_int_wtCam, len(flow_class_prediction), flow_int_attention, thr=config.TEST.TEMPSEG_LIST_THR_FLOW)
                    # Temporal Proposal
                    flow_temp_prop = get_temp_proposal(flow_temp_idx, flow_int_wtCam, flow_int_attention, flow_class_prediction,
                                                       scale, vid_len)


            else:
                # stream RGB
                if test_dataset_rgb is not None:
                    rgb_features = test_dataset_rgb.__getitem__(i, sampling_method='uniform')[0]
                    rgb_features = torch.Tensor(rgb_features).cuda()
                    vid_len = test_dataset_rgb.get_video_len(i)

                    # forward propagation
                    rgb_attention = model_rgb.att_head(rgb_features)
                    # rgb_features_fg = (rgb_features * rgb_attention).mean(dim=0, keepdim=True)
                    rgb_features_fg = (rgb_features * rgb_attention).sum(dim=0, keepdim=True) / rgb_attention.sum(dim=0, keepdim=True)
                    rgb_class_result = F.softmax(model_rgb.clf_head(rgb_features_fg))[:, 0: config.DATASET.CLF_DIM - 1]
                    rgb_class_w = model_rgb.clf_head.fc1.weight[0: config.DATASET.CLF_DIM - 1]

                    # convert to numpy
                    rgb_features = rgb_features.cpu().numpy()
                    rgb_attention = rgb_attention.cpu().numpy()
                    rgb_class_result = rgb_class_result.cpu().numpy()
                    rgb_class_w = rgb_class_w.cpu().numpy().transpose()

                    # Gathering Classification Result
                    if rgb_class_result.max() > 0.1:
                        rgb_class_prediction = np.where(rgb_class_result > class_threshold)[1]
                    else:
                        rgb_class_prediction = np.array(np.argmax(rgb_class_result), dtype=np.int).reshape(-1)

                    # get tCAM
                    rgb_tCam = get_tCAM(rgb_features, rgb_class_w)

                    # generate proposals
                    if rgb_class_prediction.any():
                        r_check = True
                        # Weighted T-CAM
                        rgb_wtCam = get_wtCAM(rgb_tCam, rgb_attention, rgb_class_prediction)
                        # Interpolate W-TCAM
                        rgb_int_wtCam = interpolated_wtCAM(rgb_wtCam, scale)
                        rgb_int_attention = upgrade_resolution(rgb_attention, scale)
                        # Get segment list of rgb_int_wtCam
                        rgb_temp_idx = get_tempseg_list(rgb_int_wtCam, len(rgb_class_prediction), rgb_int_attention, thr=config.TEST.TEMPSEG_LIST_THR_RGB)
                        # Temporal Proposal
                        rgb_temp_prop = get_temp_proposal(rgb_temp_idx, rgb_int_wtCam, rgb_class_prediction,
                                                                scale, vid_len)

                # stream FLOW
                if test_dataset_flow is not None:
                    flow_features = test_dataset_flow.__getitem__(i, sampling_method='uniform')[0]
                    flow_features = torch.Tensor(flow_features).cuda()
                    vid_len = test_dataset_flow.get_video_len(i)

                    # forward propagation
                    flow_attention = model_flow.att_head(flow_features)
                    # flow_features_fg = (flow_features * flow_attention).mean(dim=0, keepdim=True)
                    flow_features_fg = (flow_features * flow_attention).sum(dim=0, keepdim=True) / flow_attention.sum(dim=0, keepdim=True)
                    flow_class_result = F.softmax(model_flow.clf_head(flow_features_fg))[:,
                                        0: config.DATASET.CLF_DIM - 1]
                    flow_class_w = model_flow.clf_head.fc1.weight[0: config.DATASET.CLF_DIM - 1]

                    # convert to numpy
                    flow_features = flow_features.cpu().numpy()
                    flow_attention = flow_attention.cpu().numpy()
                    flow_class_result = flow_class_result.cpu().numpy()
                    flow_class_w = flow_class_w.cpu().numpy().transpose()

                    # Gathering Classification Result
                    if flow_class_result.max() > 0.1:
                        flow_class_prediction = np.where(flow_class_result > class_threshold)[1]
                    else:
                        flow_class_prediction = np.array(np.argmax(flow_class_result), dtype=np.int).reshape(-1)

                    # get tCAM
                    flow_tCam = get_tCAM(flow_features, flow_class_w)

                    # generate proposals
                    if flow_class_prediction.any():
                        f_check = True
                        # Weighted T-CAM
                        flow_wtCam = get_wtCAM(flow_tCam, flow_attention, flow_class_prediction)
                        # Interpolate W-TCAM
                        flow_int_wtCam = interpolated_wtCAM(flow_wtCam, scale)
                        flow_int_attention = upgrade_resolution(flow_attention, scale)
                        # Get segment list of flow_int_wtCam
                        flow_temp_idx = get_tempseg_list(flow_int_wtCam, len(flow_class_prediction), flow_int_attention, thr=config.TEST.TEMPSEG_LIST_THR_FLOW)
                        # Temporal Proposal
                        flow_temp_prop = get_temp_proposal(flow_temp_idx, flow_int_wtCam, flow_class_prediction,
                                                          scale, vid_len)

            # Fuse two stream and perform non-maximum suppression
            t_factor = (NUM_INPUT_FRAMES * vid_len) / (scale * NUM_SEGMENTS * SAMPLING_FRAMES)
            final_result.append(dict())

            if r_check and f_check:
                temp_prop = integrated_prop(rgb_temp_prop, flow_temp_prop, list(rgb_class_prediction),
                                                  list(flow_class_prediction))
                final_result[i]['if_two_streams'] = True
                final_result[i]['vid_name'] = vid_name
                final_result[i]['prop'] = temp_prop
                final_result[i]['att'] = 0.5 * rgb_int_attention + 0.5 * flow_int_attention
                final_result[i]['tCAM'] = 0.5 * upgrade_resolution(rgb_tCam, scale) + 0.5 * upgrade_resolution(flow_tCam, scale)
                final_result[i]['tCAM'] /= final_result[i]['tCAM'].max()
                final_result[i]['t_factor'] = t_factor
                final_result[i]['gt'] = segments[i]
            elif r_check and not f_check:
                final_result[i]['if_two_streams'] = False
                final_result[i]['vid_name'] = vid_name
                final_result[i]['prop'] = rgb_temp_prop
                final_result[i]['att'] = rgb_int_attention
                final_result[i]['tCAM'] = upgrade_resolution(rgb_tCam, scale)
                final_result[i]['tCAM'] /= final_result[i]['tCAM'].max()
                final_result[i]['t_factor'] = t_factor
                final_result[i]['gt'] = segments[i]
            elif not r_check and f_check:
                final_result[i]['if_two_streams'] = False
                final_result[i]['vid_name'] = vid_name
                final_result[i]['prop'] = flow_temp_prop
                final_result[i]['att'] = flow_int_attention
                final_result[i]['tCAM'] = upgrade_resolution(flow_tCam, scale)
                final_result[i]['tCAM'] /= final_result[i]['tCAM'].max()
                final_result[i]['t_factor'] = t_factor
                final_result[i]['gt'] = segments[i]
            else:
                final_result[i]['if_two_streams'] = False

            inf_progress(i, config.DATASET.VIDEONUM_TEST, 'Progress', 'Complete', 1, 50)


    print('\n')

    test_vid_list.close()

    return final_result


def plot_result(result_new, result_baseline):
    len = result_new['att'].shape[0]
    t_factor = result_new['t_factor']

    x = []
    y = []
    model = []  # ours / baseline / gt
    metrics = []  # gt / att / cls / loc

    # ground truth
    if_fg = np.zeros((len,))
    print(result_new['gt'])
    for segment in result_new['gt']:
        if_fg[int(segment[0] / t_factor): int(segment[1] / t_factor)] = 1

    x += range(len)
    y += list(if_fg)
    model += ['gt' for i in range(len)]
    metrics += ['gt' for i in range(len)]


    # attention
    x += range(len)
    y += list(result_new['att'].flatten())
    model += ['ours' for i in range(len)]
    metrics += ['att' for i in range(len)]

    x += range(len)
    y += list(result_baseline['att'].flatten())
    model += ['baseline' for i in range(len)]
    metrics += ['att' for i in range(len)]


    # tCAM
    x += range(len)
    y += list(result_new['tCAM'].max(axis=1).flatten())
    model += ['ours' for i in range(len)]
    metrics += ['cls' for i in range(len)]

    x += range(len)
    y += list(result_baseline['tCAM'].max(axis=1).flatten())
    model += ['baseline' for i in range(len)]
    metrics += ['cls' for i in range(len)]


    # localization
    loc_new = np.zeros((len,))
    result_new['prop'] = sorted(result_new['prop'], key=lambda x: x[1], reverse=True)
    for prop in result_new['prop'][:40]:
        if prop[1] < 0.1:  # discard proposals with score less than 0.1
            continue
        loc_new[int(prop[2] / t_factor): int(prop[3] / t_factor)] = np.maximum(loc_new[int(prop[2] / t_factor): int(prop[3] / t_factor)], prop[1])
    x += range(len)
    y += list(loc_new)
    model += ['ours' for i in range(len)]
    metrics += ['loc' for i in range(len)]

    loc_baseline = np.zeros((len,))
    result_baseline['prop'] = sorted(result_baseline['prop'], key=lambda x: x[1], reverse=True)
    for prop in result_baseline['prop'][:40]:
        if prop[1] < 0.1:  # discard proposals with score less than 0.1
            continue
        loc_baseline[int(prop[2] / t_factor): int(prop[3] / t_factor)] = np.maximum(loc_baseline[int(prop[2] / t_factor): int(prop[3] / t_factor)], prop[1])
    x += range(len)
    y += list(loc_baseline)
    model += ['baseline' for i in range(len)]
    metrics += ['loc' for i in range(len)]


    # calculate the ratio of removed context / ground truth
    if np.count_nonzero(if_fg) > 0:
        # ratio = np.count_nonzero(((loc_baseline - loc_new) > 0.4).astype(np.float) * (1 - if_fg)) / np.count_nonzero(if_fg)
        ratio = np.count_nonzero((((loc_baseline > 0).astype(np.float) - (loc_new > 0).astype(np.float)) > 0).astype(np.float) * (1 - if_fg)) / len
        print('ratio of removed context / ground truth is {}'.format(ratio))

    data = pd.DataFrame(dict(x = x, y = y, model = model, metrics = metrics))
    g = sns.relplot(x = 'x', y = 'y', row = 'metrics', hue = 'model', kind = 'line', data = data, aspect=10)
    axes = g.axes.flatten()
    axes[0].set_title("Ground Truth")
    axes[0].set_ylabel('')
    axes[0].set_ylim(-0.5, 1.5)
    axes[1].set_title("Attention")
    axes[1].set_ylabel('')
    axes[1].set_ylim(-0.5, 1.5)
    axes[2].set_title("Classification")
    axes[2].set_ylabel('')
    axes[2].set_ylim(-0.5, 1.5)
    axes[3].set_title("Localization")
    axes[3].set_xlabel('frame')
    axes[3].set_ylabel('')
    axes[3].set_ylim(-0.5, 1.5)
    return g



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

    model_flow = create_model()
    model_flow.my_load_state_dict(torch.load(config.TEST.STATE_DICT_FLOW), strict=True)
    model_flow = model_flow.cuda(gpus[0])

    baseline_rgb = create_model()
    baseline_rgb.my_load_state_dict(torch.load(os.path.join('experiments/', config.DATASET_NAME, 'train/final_rgb_0.27497961238219276.pth')))
    baseline_rgb = baseline_rgb.cuda(gpus[0])

    baseline_flow = create_model()
    baseline_flow.my_load_state_dict(torch.load(os.path.join('experiments/', config.DATASET_NAME, 'train/final_flow_0.27497961238219276.pth')))
    baseline_flow = baseline_flow.cuda(gpus[0])

    # load data
    test_dataset_rgb = get_dataset(mode='test', modality='rgb')
    test_dataset_flow = get_dataset(mode='test', modality='flow')


    final_results = test(test_dataset_rgb, model_rgb, test_dataset_flow, model_flow)
    final_results_baseline = test(test_dataset_rgb, baseline_rgb, test_dataset_flow, baseline_flow)

    save_dir = os.path.join('experiments/', config.DATASET_NAME, 'visualization')
    for i in range(len(final_results)):
        # if i is not 21:
        #     continue
        if not final_results[i]['if_two_streams'] or not final_results_baseline[i]['if_two_streams']:
            continue
        print(i, final_results[i]['vid_name'])
        g = plot_result(final_results[i], final_results_baseline[i])
        g.savefig(os.path.join(save_dir, final_results[i]['vid_name']))


if __name__ == '__main__':
    main()