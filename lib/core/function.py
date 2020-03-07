from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os
import json
import random

import numpy as np
import scipy as sp
import torch
import torch.nn.functional as F

import _init_paths
from core.config import config
from core.loss import loss_fg, loss_bg, loss_sparse, loss_guide, loss_cluster, loss_ising, loss_cvae
from utils.utils import AverageMeter
from utils.utils import get_tCAM, get_wtCAM, upgrade_resolution, interpolated_wtCAM, get_tempseg_list, get_temp_proposal
from utils.utils import integrated_prop, result2json, json2txt, inf_progress
from utils.utils import bar_figure


logger = logging.getLogger(__name__)

cvae_sample_num = 1  # how many times to sample latent variables


def train(train_loader, model, cvae, optimizer, optimizer_cvae, epoch, epoch_num, modality=None):
    config.DATASET.NUM_SEGMENTS = 400

    # build recorders
    batch_time = AverageMeter()
    data_time = AverageMeter()
    clf_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    train_list = list((train_loader))

    for k in range(epoch_num):

        # freeze the model
        for name, param in model.named_parameters():
            param.requires_grad = False
            param.grad = None

        # training of vae
        for hhh in range(1):# * ((k + 1) % 5 == 0)):
            random.shuffle(train_list)
            for i, (video_feature, label) in enumerate(train_list):
                # measure data loading time
                data_time.update(time.time() - end)

                original_shape = video_feature.size()
                total_batch_size = label.shape[0]

                video_feature = video_feature.cuda()
                label = label.type(torch.float).cuda()

                # compute the attention
                attention = model(video_feature, 'att')  # [batch_size, seg_num, 1]

                # # compute previous feature and attention
                # condition = attention.clone()
                # for pre_idx in range(1, config.MODEL.CONDITION_FRAME_NUM + 1):
                #     video_feature_pre = torch.zeros_like(video_feature, device='cuda')
                #     video_feature_pre[:, pre_idx:, :] = video_feature[:, :-pre_idx, :]
                #     attention_pre = torch.zeros_like(attention, device='cuda')
                #     attention_pre[:, pre_idx:, :] = attention[:, :-pre_idx, :]
                #     condition = torch.cat([condition, video_feature_pre, attention_pre], dim=-1)

                loss = 0
                for l in range(cvae_sample_num):
                    # forward through cvae
                    means, log_var, z, recon_feature = cvae('forward', video_feature, attention)

                    # compute loss of cvae
                    loss += loss_cvae(recon_feature, video_feature, means, log_var, attention)

                loss /= cvae_sample_num
                loss *= (10 * min(epoch, 300) / 300)

                # back prop
                optimizer_cvae.zero_grad()
                loss.backward()
                optimizer_cvae.step()

                # update total clf_loss
                clf_losses.update(loss.item(), total_batch_size)

                # update time record
                batch_time.update(time.time() - end)
                end = time.time()

                # logging
                if i % config.TRAIN.PRINT_EVERY_STEP == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                        epoch + k, i, len(train_loader), batch_time=batch_time,
                        speed=total_batch_size / batch_time.val,
                        data_time=data_time, loss=clf_losses)
                    logger.info(msg)

        # wake up the model
        for name, param in model.named_parameters():
            param.requires_grad = True
            param.grad = None

        # freeze cvae
        for name, param in cvae.named_parameters():
            param.requires_grad = False
            param.grad = None

        # training of model
        for hhh in range(1):
            random.shuffle(train_list)
            for i, (video_feature, label) in enumerate(train_list):
                # measure data loading time
                data_time.update(time.time() - end)

                original_shape = video_feature.size()
                total_batch_size = label.shape[0]

                video_feature = video_feature.cuda()
                label = label.type(torch.float).cuda()


                # compute the attention
                attention = model(video_feature, 'att')  # [batch_size, seg_num, 1]

                # compute the foreground / background feature -> [batch_size, feature_dim]
                feature_fg = (video_feature * attention).sum(dim=1) / (attention.sum(dim=1) + 1)
                feature_bg = (video_feature * (1 - attention)).sum(dim=1) / ((1 - attention).sum(dim=1) + 1e-6)

                # compute foreground / background clf score (not softmaxed)
                clf_score_fg = model(feature_fg, 'clf')
                clf_score_bg = model(feature_bg, 'clf')

                # compute forground / background cluster gap for loss_cluster
                gap_fg, gap_bg = model([feature_fg, feature_bg], mode='cluster')

                # # compute previous feature and attention
                # condition = attention.clone()
                # for pre_idx in range(1, config.MODEL.CONDITION_FRAME_NUM + 1):
                #     video_feature_pre = torch.zeros_like(video_feature, device='cuda')
                #     video_feature_pre[:, pre_idx:, :] = video_feature[:, :-pre_idx, :]
                #     attention_pre = torch.zeros_like(attention, device='cuda')
                #     attention_pre[:, pre_idx:, :] = attention[:, :-pre_idx, :]
                #     condition = torch.cat([condition, video_feature_pre, attention_pre], dim=-1)


                # compute loss
                l_fg = loss_fg(clf_score_fg, label)
                l_bg = loss_bg(clf_score_bg)
                l_sparse = loss_sparse(attention)
                l_guide = loss_guide(video_feature, label, attention.view(total_batch_size, config.DATASET.NUM_SEGMENTS),
                                     model.module.clf_head.fc1.weight.data.transpose(0, 1))
                l_cluster = loss_cluster(gap_fg, gap_bg)
                l_ising = loss_ising(attention)
                l_recon = 0
                for l in range(cvae_sample_num):
                    recon_feature = cvae('inference', att=attention)
                    l_recon += (recon_feature - video_feature).pow(2).mean()
                l_recon /= cvae_sample_num

                logger.info(l_fg)
                logger.info(l_bg)
                logger.info(l_sparse)
                logger.info(l_guide)
                logger.info(l_cluster)
                logger.info(l_recon)
                logger.info(l_ising)

                if config.DATASET_NAME == 'THUMOS14':
                    if modality == 'rgb':
                        loss2 = 1.5 * l_fg + 0.03 * l_bg + 0.1 * l_guide + min(epoch, 300) / 300 * 0.5 * l_recon
                    elif modality == 'flow':
                        loss2 = 1 * l_fg + 0.03 * l_bg + 0.1 * l_guide + min(epoch, 300) / 300 * 0.3 * l_recon
                elif config.DATASET_NAME == 'ActivityNet12':
                    if modality == 'rgb':
                        loss2 = 1 * l_fg + 1 * l_bg + 0.1 * l_guide + 0.1 * l_recon
                    elif modality == 'flow':
                        loss2 = 0.3 * l_fg + 1 * l_bg + 0.1 * l_guide + 0.1 * l_recon

                # back prop
                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()

                # update total clf_loss
                clf_losses.update(loss2.item(), total_batch_size)

                # update time record
                batch_time.update(time.time() - end)
                end = time.time()

                # logging
                if i % config.TRAIN.PRINT_EVERY_STEP == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                        epoch + k, i, len(train_loader), batch_time=batch_time,
                        speed=total_batch_size / batch_time.val,
                        data_time=data_time, loss=clf_losses)
                    logger.info(msg)

        # wake up cvae
        for name, param in cvae.named_parameters():
            param.requires_grad = True
            param.grad = None


def test(test_loader, model, epoch):
    # FIXME: to be updated
    # build recorders
    batch_time = AverageMeter()
    clf_losses = AverageMeter()

    # switch to train mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (video_feature, label) in enumerate(test_loader):
            original_shape = video_feature.size()
            total_batch_size = label.shape[0]

            video_feature = video_feature.cuda()
            label = label.type(torch.float).cuda()

            # compute the attention
            attention = model(video_feature, 'att')  # [batch_size, seg_num, 1]

            # compute the foreground / background feature -> [batch_size, feature_dim]
            feature_fg = (video_feature * attention).mean(dim=1)
            feature_bg = (video_feature * (1 - attention)).mean(dim=1)

            # compute foreground / background clf score (not softmaxed)
            clf_score_fg = model(feature_fg, 'clf')
            clf_score_bg = model(feature_bg, 'clf')

            # compute loss
            loss = loss_fg(clf_score_fg, label) + loss_bg(clf_score_bg)

            # update total clf_loss
            clf_losses.update(loss.item(), total_batch_size)

            # update time record
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TEST.PRINT_EVERY_STEP == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    i, len(test_loader), batch_time=batch_time,
                    loss=clf_losses)
                logger.info(msg)

    return -clf_losses.avg


def test_final(test_dataset_rgb, model_rgb, test_dataset_flow, model_flow, tb_writer=None):
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

            if tb_writer is not None:
                if i < 10:
                    if test_dataset_rgb is not None:
                        # tb_writer.add_figure('att_{}_rgb'.format(i), bar_figure(rgb_attention.reshape((-1))))
                        # tb_writer.add_figure('tCAM_{}_rgb'.format(i),
                        #                      bar_figure(sp.special.expit(rgb_tCam).max(axis=1).reshape((-1))))
                        tb_writer.add_figure('wtCAM_{}_rgb'.format(i),
                                             bar_figure(rgb_wtCam[:, :, 0].max(axis=1)))
                    if test_dataset_flow is not None:
                        # tb_writer.add_figure('att_{}_flow'.format(i), bar_figure(flow_attention.reshape((-1))))
                        # tb_writer.add_figure('tCAM_{}_flow'.format(i),
                        #                      bar_figure(sp.special.expit(flow_tCam).max(axis=1).reshape((-1))))
                        tb_writer.add_figure('wtCAM_{}_flow'.format(i),
                                             bar_figure(flow_wtCam[:, :, 0].max(axis=1)))

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


def train_adv(train_loader, model, optimizer, epoch, epoch_num):
    # build recorders
    batch_time = AverageMeter()
    data_time = AverageMeter()
    clf_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    train_list = list((train_loader))

    # training of f_2
    for k in range(epoch_num):
        random.shuffle(train_list)
        for i, (video_feature, label) in enumerate(train_list):
            # measure data loading time
            data_time.update(time.time() - end)

            original_shape = video_feature.size()
            total_batch_size = label.shape[0]

            video_feature = video_feature.cuda()
            label = label.type(torch.float).cuda()


            # compute the attention
            attention = model(video_feature, 'att').detach()  # [batch_size, seg_num, 1]

            # compute the foreground / background feature -> [batch_size, feature_dim]
            feature_fg = (video_feature * attention).sum(dim=1) / (attention.sum(dim=1) + 1)
            feature_bg = (video_feature * (1 - attention)).sum(dim=1) / ((1 - attention).sum(dim=1) + 1e-6)

            # compute foreground / background clf score (not softmaxed)
            clf_score_fg = model(feature_fg, 'clf')
            clf_score_bg = model(feature_bg, 'clf')

            # compute forground / background cluster gap for loss_cluster
            gap_fg, gap_bg = model([feature_fg, feature_bg], mode='cluster')

            # compute loss
            # l_bg_1 = loss_bg(clf_score_bg)
            l_bg_2 = loss_fg(clf_score_bg, label)
            l_fg = loss_fg(clf_score_fg, label)
            l_bg = loss_bg(clf_score_bg)
            l_sparse = loss_sparse(attention)
            l_guide = loss_guide(video_feature, label, attention.view(total_batch_size, config.DATASET.NUM_SEGMENTS),
                                 model.module.clf_head.fc1.weight.data.transpose(0, 1))
            l_cluster = loss_cluster(gap_fg, gap_bg)
            logger.info(l_fg)
            logger.info(l_bg)
            logger.info(l_sparse)
            logger.info(l_guide)
            logger.info(l_cluster)
            loss = l_fg + 1 * l_bg


            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update total clf_loss
            clf_losses.update(loss.item(), total_batch_size)

            # update time record
            batch_time.update(time.time() - end)
            end = time.time()

            # logging
            if i % config.TRAIN.PRINT_EVERY_STEP == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                    epoch + k, i, len(train_loader), batch_time=batch_time,
                    speed=total_batch_size / batch_time.val,
                    data_time=data_time, loss=clf_losses)
                logger.info(msg)

    # freeze clf_head
    for param in model.module.clf_head.parameters():
        param.requires_grad = False

    # training of f_1
    for k in range(epoch_num):
        random.shuffle(train_list)
        for i, (video_feature, label) in enumerate(train_list):

            # measure data loading time
            data_time.update(time.time() - end)

            original_shape = video_feature.size()
            total_batch_size = label.shape[0]

            video_feature = video_feature.cuda()
            label = label.type(torch.float).cuda()


            # compute the attention
            attention = model(video_feature, 'att')  # [batch_size, seg_num, 1]

            # compute the foreground / background feature -> [batch_size, feature_dim]
            feature_fg = (video_feature * attention).sum(dim=1) / (attention.sum(dim=1) + 1)
            feature_bg = (video_feature * (1 - attention)).sum(dim=1) / ((1 - attention).sum(dim=1) + 1e-6)

            # compute foreground / background clf score (not softmaxed)
            clf_score_fg = model(feature_fg, 'clf')
            clf_score_bg = model(feature_bg, 'clf')

            # compute forground / background cluster gap for loss_cluster
            gap_fg, gap_bg = model([feature_fg, feature_bg], mode='cluster')


            # compute loss
            # l_bg_1 = loss_bg(clf_score_bg)
            # l_bg_2 = loss_fg(clf_score_bg, label)
            l_fg = loss_fg(clf_score_fg, label)
            l_bg = loss_bg(clf_score_bg)
            l_sparse = loss_sparse(attention)
            l_guide = loss_guide(video_feature, label, attention.view(total_batch_size, config.DATASET.NUM_SEGMENTS),
                                 model.module.clf_head.fc1.weight.data.transpose(0, 1))
            l_cluster = loss_cluster(gap_fg, gap_bg)
            logger.info(l_fg)
            logger.info(l_bg)
            logger.info(l_sparse)
            logger.info(l_guide)
            logger.info(l_cluster)
            loss = l_fg + 0.03 * l_bg + 0.1 * l_guide + 0.3 * l_cluster


            # back prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update total clf_loss
            clf_losses.update(loss.item(), total_batch_size)

            # update time record
            batch_time.update(time.time() - end)
            end = time.time()

            # logging
            if i % config.TRAIN.PRINT_EVERY_STEP == 0:
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                    epoch + k, i, len(train_loader), batch_time=batch_time,
                    speed=total_batch_size / batch_time.val,
                    data_time=data_time, loss=clf_losses)
                logger.info(msg)

    # wake up clf_head
    for param in model.module.clf_head.parameters():
        param.requires_grad = True