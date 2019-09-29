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
from core.function import train
from core.function import train_adv
from core.function import test
from core.function import test_final
from dataset.dataset import get_dataset
from models.model import create_model, create_cvae
from utils.utils import create_optimizer
from utils.utils import create_logger
from eval.get_detection_performance import eval_mAP


def main():
    # convert to train mode
    config.MODE = 'train'
    extra()

    # create a logger
    logger = create_logger(config, 'train')

    # logging configurations
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # create a model
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    gpus = [int(i) for i in config.GPUS.split(',')]
    # gpus = range(gpus.__len__())

    model_rgb = create_model()
    if config.TRAIN.RESUME_RGB:
        model_rgb.my_load_state_dict(torch.load(config.TRAIN.STATE_DICT_RGB), strict=True)

    model_rgb = model_rgb.cuda(gpus[0])
    model_rgb = torch.nn.DataParallel(model_rgb, device_ids=gpus)

    model_flow = create_model()
    if config.TRAIN.RESUME_FLOW:
        model_flow.my_load_state_dict(torch.load(config.TRAIN.STATE_DICT_FLOW), strict=True)

    model_flow = model_flow.cuda(gpus[0])
    model_flow = torch.nn.DataParallel(model_flow, device_ids=gpus)

    # create a conditional-vae
    cvae_rgb = create_cvae()
    cvae_rgb = cvae_rgb.cuda(gpus[0])
    cvae_rgb = torch.nn.DataParallel(cvae_rgb, device_ids=gpus)

    cvae_flow = create_cvae()
    cvae_flow = cvae_flow.cuda(gpus[0])
    cvae_flow = torch.nn.DataParallel(cvae_flow, device_ids=gpus)

    # create an optimizer
    optimizer_rgb = create_optimizer(config, model_rgb)
    optimizer_flow = create_optimizer(config, model_flow)
    optimizer_cvae_rgb = create_optimizer(config, cvae_rgb)
    optimizer_cvae_flow = create_optimizer(config, cvae_flow)

    # create a learning rate scheduler
    lr_scheduler_rgb = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_rgb, mode='max', factor=0.3, patience=500 // config.TRAIN.TEST_EVERY_EPOCH,
        verbose=True
    )
    lr_scheduler_flow = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_flow, mode='max', factor=0.3, patience=500 // config.TRAIN.TEST_EVERY_EPOCH,
        verbose=True
    )
    lr_scheduler_cvae_rgb = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_cvae_rgb, config.TRAIN.LR_MILESTONES,
        config.TRAIN.LR_DECAY_RATE
    )
    lr_scheduler_cvae_flow = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_cvae_flow, config.TRAIN.LR_MILESTONES,
        config.TRAIN.LR_DECAY_RATE
    )

    # load data
    train_dataset_rgb = get_dataset(mode='train', modality='rgb')
    train_dataset_flow = get_dataset(mode='train', modality='flow')
    test_dataset_rgb = get_dataset(mode='test', modality='rgb')
    test_dataset_flow = get_dataset(mode='test', modality='flow')

    train_loader_rgb = torch.utils.data.DataLoader(
        train_dataset_rgb,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
    train_loader_flow = torch.utils.data.DataLoader(
        train_dataset_flow,
        batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True,
        drop_last=True
    )
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

    # training and validating

    best_perf = 0

    best_model_rgb = create_model()
    best_model_rgb = best_model_rgb.cuda(gpus[0])

    best_model_flow = create_model()
    best_model_flow = best_model_flow.cuda(gpus[0])

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH, config.TRAIN.TEST_EVERY_EPOCH):
        # train rgb for **config.TRAIN.TEST_EVERY_EPOCH** epochs
        train(train_loader_rgb, model_rgb, cvae_rgb, optimizer_rgb, optimizer_cvae_rgb, epoch,
              config.TRAIN.TEST_EVERY_EPOCH, 'rgb')

        # evaluate on validation set
        result_file_path_rgb = test_final(test_dataset_rgb, model_rgb.module, test_dataset_flow, best_model_flow)
        perf_indicator = eval_mAP(config.DATASET.GT_JSON_PATH, result_file_path_rgb)

        if best_perf < perf_indicator:
            logger.info("(rgb) new best perf: {:3f}".format(perf_indicator))
            best_perf = perf_indicator
            best_model_rgb.my_load_state_dict(model_rgb.state_dict(), strict=True)

        # lr_scheduler_rgb.step(perf_indicator_rgb)
        # lr_scheduler_cvae_rgb.step()


        # train flow for **config.TRAIN.TEST_EVERY_EPOCH** epochs
        train(train_loader_flow, model_flow, cvae_flow, optimizer_flow, optimizer_cvae_flow, epoch,
              config.TRAIN.TEST_EVERY_EPOCH, 'flow')

        # evaluate on validation set
        result_file_path_flow = test_final(test_dataset_rgb, best_model_rgb, test_dataset_flow, model_flow.module)
        perf_indicator = eval_mAP(config.DATASET.GT_JSON_PATH, result_file_path_flow)

        if best_perf < perf_indicator:
            logger.info("(flow) new best perf: {:3f}".format(perf_indicator))
            best_perf = perf_indicator
            best_model_flow.my_load_state_dict(model_flow.state_dict(), strict=True)

        # lr_scheduler_flow.step(perf_indicator_flow)
        # lr_scheduler_cvae_flow.step()

    logger.info("=> saving final result into {}".format(
        os.path.join(config.OUTPUT_DIR, 'final_rgb_{}.pth'.format(best_perf))))
    torch.save(best_model_rgb.state_dict(),
               os.path.join(config.OUTPUT_DIR, 'final_rgb_{}.pth'.format(best_perf)))

    logger.info("=> saving final result into {}".format(
        os.path.join(config.OUTPUT_DIR, 'final_flow_{}.pth'.format(best_perf))))
    torch.save(best_model_flow.state_dict(),
               os.path.join(config.OUTPUT_DIR, 'final_flow_{}.pth'.format(best_perf)))




if __name__ == '__main__':
    main()



