from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

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
from eval.get_detection_performance import eval_mAP

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

    result_file_path = test_final(test_dataset_rgb, model_rgb, test_dataset_flow, model_flow)
    eval_mAP(config.DATASET.GT_JSON_PATH, result_file_path)
    # test_final(None, None, test_dataset_flow, model_flow)
    # test_final(test_dataset_rgb, model_rgb, None, None)


if __name__ == '__main__':
    main()