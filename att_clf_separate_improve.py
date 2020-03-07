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

import _init_paths
from core.config import config
from core.config import extra
from dataset.dataset import get_dataset
from models.model import create_model
from utils.utils import create_logger
from utils.utils import get_tCAM, get_wtCAM, upgrade_resolution, interpolated_wtCAM, get_tempseg_list, get_temp_proposal
from utils.utils import integrated_prop, result2json, json2txt, inf_progress
from utils.utils import bar_figure


logger = logging.getLogger(__name__)

def att_clf_separate_test(test_dataset_rgb, model_rgb, baseline_rgb,
                          test_dataset_flow, model_flow, baseline_flow):
    # some hyper-parameters
    scale = config.TEST.SCALE
    class_threshold = config.TEST.CLASS_THRESHOLD

    # file for matching 'video number' and 'video name'
    test_vid_list = open(os.path.join(config.DATASET.DATAROOT, 'test_data',
                                      config.DATASET.TEST_VID_LIST), 'r')
    lines = test_vid_list.read().splitlines()

    # Define json File (output)
    final_result = dict()
    final_result['version'] = 'VERSION 1.3'
    final_result['results'] = {}
    final_result['external_data'] = {'used': True, 'details': 'Features from I3D Net'}

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
                rgb_attention = baseline_rgb.att_head(rgb_features)
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
                flow_attention = baseline_flow.att_head(flow_features)
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



                    # rgb_temp_prop = None
                    # for thr in np.arange(0.025, 0.5, 0.025):
                    #     # Get segment list of rgb_int_wtCam
                    #     rgb_temp_idx = get_tempseg_list(rgb_int_wtCam, len(rgb_class_prediction), rgb_int_attention, thr)
                    #     # Temporal Proposal
                    #     rgb_temp2_prop = get_temp_proposal(rgb_temp_idx, rgb_int_wtCam, rgb_class_prediction,
                    #                                       scale, vid_len)
                    #
                    #     if rgb_temp_prop is None:
                    #         rgb_temp_prop = rgb_temp2_prop
                    #     else:
                    #         for k in range(rgb_temp_prop.__len__()):
                    #             rgb_temp_prop[k] = [list(obj) for obj in integrated_prop([rgb_temp_prop[k]], [rgb_temp2_prop[k]],
                    #                                         [list(rgb_class_prediction)[k]],
                    #                                         [list(rgb_class_prediction)[k]])]

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
            if r_check and f_check:
                temp_prop = integrated_prop(rgb_temp_prop, flow_temp_prop, list(rgb_class_prediction),
                                                  list(flow_class_prediction))
                final_result['results'][vid_name] = result2json([temp_prop])
            elif r_check and not f_check:
                final_result['results'][vid_name] = result2json(rgb_temp_prop)
            elif not r_check and f_check:
                final_result['results'][vid_name] = result2json(flow_temp_prop)

            inf_progress(i, config.DATASET.VIDEONUM_TEST, 'Progress', 'Complete', 1, 50)


    print('\n')

    # Save Results
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    json_path = os.path.join(config.OUTPUT_DIR, 'results_{}_{}_{}.json'.format(config.DATASET_NAME, time_str, os.getpid()))
    logger.info("saving detection json results in {}".format(json_path))
    with open(json_path, 'w') as fp:
        json.dump(final_result, fp)

    txt_path = os.path.join(config.OUTPUT_DIR, 'results_{}_{}.txt'.format(config.DATASET_NAME, time_str))
    logger.info("saving detection txt results in {}".format(txt_path))
    # with open(txt_path, 'w') as tp:
    #     json2txt(final_result['results'], tp)

    test_vid_list.close()

    return json_path


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

    baseline_rgb = create_model()
    baseline_rgb.my_load_state_dict(torch.load(os.path.join('experiments/', config.DATASET_NAME, 'train/final_rgb_0.27497961238219276.pth')))
    baseline_rgb = baseline_rgb.cuda(gpus[0])

    baseline_flow = create_model()
    baseline_flow.my_load_state_dict(torch.load(os.path.join('experiments/', config.DATASET_NAME, 'train/final_flow_0.27497961238219276.pth')))
    baseline_flow = baseline_flow.cuda(gpus[0])

    # load data
    test_dataset_rgb = get_dataset(mode='test', modality='rgb')
    test_dataset_flow = get_dataset(mode='test', modality='flow')

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

    att_clf_separate_test(test_dataset_rgb, model_rgb, baseline_rgb,
                          test_dataset_flow, model_flow, baseline_flow)
    # test_final(None, None, test_dataset_flow, model_flow)
    # test_final(test_dataset_rgb, model_rgb, None, None)


if __name__ == '__main__':
    main()