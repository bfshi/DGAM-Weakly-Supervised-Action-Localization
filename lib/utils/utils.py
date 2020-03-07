from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
from pathlib import Path
import json
import sys
import math

import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.ops import nms
from torch._six import container_abcs, string_classes, int_classes

import _init_paths
from core.config import config

NUM_SEGMENTS = config.DATASET.NUM_SEGMENTS
SAMPLING_FRAMES = config.DATASET.SAMPLING_FRAMES
NUM_INPUT_FRAMES = config.DATASET.NUM_INPUT_FRAMES

CLASS = config.DATASET.CLASS
CLASS_INDEX = config.DATASET.CLASS_INDEX

def create_optimizer(cfg, model):
    """
    create an SGD or ADAM optimizer
    :param cfg: global configs
    :param model: the model to be trained
    :return: an SGD or ADAM optimizer
    """
    optimizer = None

    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            betas=cfg.TRAIN.BETA
        )

    return optimizer


def create_logger(cfg, phase='train'):
    """
    create a logger for experiment record
    To use a logger to publish message m, just run logger.info(m)
    :param cfg: global config
    :param phase: train or val
    :return: a logger
    """
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg.DATASET_NAME, time_str, phase)
    final_log_file = Path(cfg.OUTPUT_DIR) / log_file
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def random_perturb(v_len, num_segments):
    """
    Given the length of video and sampling number, which segments should I choose?
    Random sampling is used.
    :param v_len: length of video
    :param num_segments: expected number of segments
    :return: a list of indices to sample
    """
    random_p = np.arange(num_segments) * v_len / num_segments
    for i in range(num_segments):
        if i < num_segments - 1:
            if int(random_p[i]) != int(random_p[i + 1]):
                random_p[i] = np.random.choice(range(int(random_p[i]), int(random_p[i + 1]) + 1))
            else:
                random_p[i] = int(random_p[i])
        else:
            if int(random_p[i]) < v_len - 1:
                random_p[i] = np.random.choice(range(int(random_p[i]), v_len))
            else:
                random_p[i] = int(random_p[i])
    return random_p.astype(int)


def uniform_sampling(vid_len, num_segments):
    """
    Given the length of video and sampling number, which segments should I choose?
    Uniform sampling is used.
    :param vid_len: length of video
    :param num_segments: expected number of segments
    :return: a list of indices to sample
    """
    u_sample = np.arange(num_segments) * vid_len / num_segments
    u_sample = np.floor(u_sample)
    return u_sample.astype(int)


def get_tCAM(feature, layer_Weights):
    """
    Get TCAM signal.
    :param feature: [seg_num, feature_dim]
    :param layer_Weights: weight of clf layer
    :return: tCAM
    """
    tCAM = np.matmul(feature, layer_Weights)
    return tCAM


def get_wtCAM(tCAM, attention_Weights, pred, sub_tCAM = None, alpha = None):
    """
    Get weighted TCAM and the score for each segment
    :param tCAM: [seg_num, clf_dim]
    :param attention_Weights: [seg_num, 1]
    :param pred: which classes do we predict from the video (could be multi-class)
    :param sub_tCAM: tCAM from another modality (if available)
    :param alpha: rgb : flow = alpha : 1 - alpha
    :return: weighted-tCAM
    """
    NUM_SEGMENTS = config.DATASET.NUM_SEGMENTS
    wtCAM = attention_Weights * sp.special.expit(tCAM)
    signal = np.reshape(wtCAM[:, pred], (NUM_SEGMENTS, -1, 1))
    if sub_tCAM is None:
        score = np.reshape(wtCAM[:, pred],
                           (NUM_SEGMENTS, -1, 1))
    else:
        sub_wtCAM = attention_Weights * sp.special.expit(sub_tCAM)
        score = np.reshape(attention_Weights * (alpha * wtCAM + (1 - alpha) * sub_wtCAM)[:, pred],
                           (NUM_SEGMENTS, -1, 1))
    return np.concatenate((signal, score), axis=2)


def upgrade_resolution(arr, scale):
    """
    Interpolate empty segments.
    """
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')  # linear/quadratic/cubic
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def interpolated_wtCAM(wT, scale):
    """
    Interpolate the wtCAM signals and threshold
    """
    final_wT = upgrade_resolution(wT, scale)
    return final_wT


def get_tempseg_list(wtcam, c_len, attention, thr=0.05):
    """
    Return the index where the wtcam value > 0.05
    """
    temp = []
    for i in range(c_len):
        if config.DATASET_NAME == 'THUMOS14':
            pos = np.where((0.8 * wtcam[:, i, 0] + 0.2 * attention[:, 0] * attention[:, 0]) > thr)
        elif config.DATASET_NAME == 'ActivityNet12':
            pos = np.where(gaussian_filter1d((0.8 * wtcam[:, i, 0] + 0.2 * attention[:, 0] * attention[:, 0]),
                                             sigma=25) > thr)
        temp_list = pos
        temp.append(temp_list)
    return temp


def grouping(arr):
    """
    Group the connected results
    """
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def get_temp_proposal(tList, wtcam, attention, c_pred, scale, v_len):
    """
    Get the temporal proposal
    """
    NUM_SEGMENTS = config.DATASET.NUM_SEGMENTS
    t_factor = (NUM_INPUT_FRAMES * v_len) / (scale * NUM_SEGMENTS * SAMPLING_FRAMES)  # Factor to convert segment index to actual timestamp
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)  # Get the connected parts
            for j in range(len(grouped_temp_list)):

                # Outer-Inner-Contrastive
                if grouped_temp_list[j][0] > 0:
                    left_outer_list = range(max(0, grouped_temp_list[j][0] - grouped_temp_list[j].size // 4 - 1),
                                            grouped_temp_list[j][0])
                    c_left_outer = np.mean(wtcam[left_outer_list, i, 1])
                else:
                    c_left_outer = 0

                if grouped_temp_list[j][-1] < int(wtcam.shape[0]) - 1:
                    right_outer_list = range(grouped_temp_list[j][-1] + 1,
                                             min(int(wtcam.shape[0]), grouped_temp_list[j][-1] + 1 + grouped_temp_list[j].size // 4 + 1))
                    c_right_outer = np.mean(wtcam[right_outer_list, i, 1])
                else:
                    c_right_outer = 0
                c_score = np.mean(wtcam[grouped_temp_list[j], i, 1]) - 0.3 * (c_left_outer + c_right_outer)
                # c_score = np.mean(wtcam[grouped_temp_list[j], i, 1])
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])  # Add the proposal
        temp.append(c_temp)
    return temp


def nms_prop(arr):
    """
    Perform Non-Maximum-Suppression
    """
    p = [2, 0, 1]
    idx = np.argsort(p)
    prop_tensor = (arr[:, 1:])[:, idx]
    fake_y = np.tile(np.array([0, 1]), (arr.shape[0], 1))
    box = prop_tensor[:, :2]
    score = prop_tensor[:, 2]
    box_prop = np.concatenate((fake_y, box), 1)
    p2 = [0, 2, 1, 3]
    pidx = np.argsort(p2)
    box_prop = box_prop[:, pidx]
    box_prop = torch.Tensor(box_prop)
    score = torch.Tensor(score)
    result = nms(box_prop, score, iou_threshold=0.5)
    return result.numpy().astype(np.int)


def integrated_prop(rgbProp, flowProp, rPred, fPred):
    """
    Fuse two stream & perform non-maximum suppression
    """
    temp = []
    for i in range(config.DATASET.CLF_DIM - 1):
        if (i in rPred) and (i in fPred):
            ridx = rPred.index(i)
            fidx = fPred.index(i)
            rgb_temp = rgbProp[ridx]
            flow_temp = flowProp[fidx]
            rgb_set = set([tuple(x) for x in rgb_temp])
            flow_set = set([tuple(x) for x in flow_temp])
            fuse_temp = np.array([x for x in rgb_set | flow_set])  # Gather RGB proposals and FLOW proposals together
            fuse_temp = np.sort(fuse_temp.view('f8,f8,f8,f8'), order=['f1'], axis=0).view(np.float)[::-1]

            if len(fuse_temp) > 0:
                nms_idx = nms_prop(fuse_temp)
                for k in nms_idx:
                    temp.append(fuse_temp[k])

        elif (i in rPred) and not (i in fPred):  # For the video which only has RGB Proposals
            ridx = rPred.index(i)
            rgb_temp = rgbProp[ridx]
            for j in range(len(rgb_temp)):
                temp.append(rgb_temp[j])
        elif not (i in rPred) and (i in fPred):  # For the video which only has FLOW Proposals
            fidx = fPred.index(i)
            flow_temp = flowProp[fidx]
            for j in range(len(flow_temp)):
                temp.append(flow_temp[j])
    return temp


def result2json(result):
    """
    Record the proposals to the json file
    """
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': CLASS[int(result[i][j][0])], 'score': result[i][j][1],
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
    return result_file


def json2txt(jF, rF):
    for i in jF.keys():
        for j in range(len(jF[i])):
            rF.write('{:s} {:f} {:f} {:s} {:f}\n'.format(i, jF[i][j]['segment'][0], jF[i][j]['segment'][1],
                                                        CLASS_INDEX[jF[i][j]['label']], round(jF[i][j]['score'], 6)))


def inf_progress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    visualize the process
    """
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_1D_gaussian_fitler(kernel_size=5, sigma=1):
    """
    create a 1D gaussian filter.
    Note that since 1D_Conv operates on the last dimension of a tensor,
    the target dimension needs to be exchanged to the last one.
    :param kernel_size: gaussian kernel size
    :param sigma: standard variation
    :return: gaussian filter
    """
    weight = torch.zeros((kernel_size,))
    for i in range(kernel_size):
        weight[i] = 1 / ((2 * math.pi)**0.5 * sigma) *\
                    np.exp(-(i - (kernel_size - 1) / 2)**2 / (2 * sigma**2))
    weight = weight / weight.sum()
    weight = weight.view(1, 1, kernel_size)

    gaussian_filter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                                stride=1, padding=(kernel_size - 1) // 2, bias=False)
    gaussian_filter.weight.data = weight
    gaussian_filter.weight.requires_grad = False
    gaussian_filter = gaussian_filter.cuda()

    return gaussian_filter


gaussian_filter = None

def gaussian_filtering(tensor):
    tensor = tensor.transpose(1, 2)
    shape = tensor.shape
    return gaussian_filter(tensor.reshape((-1, 1, shape[2]))).reshape(shape).transpose(1, 2)



def bar_figure(data_series):
    figure = plt.figure()
    plt.plot(range(data_series.size), data_series)
    return figure


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


