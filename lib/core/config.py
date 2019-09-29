from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import sys

import numpy as np
from easydict import EasyDict as edict



# global configuration

config = edict()



# common configs

config.GPUS = '0'
config.GPU_NUM = 1  # number of gpus in config.GPUS
config.WORKERS = 4
config.MODE = 'train'  # train / test



# Cudnn related params

config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True



# dataset related configs

config.DATASET_NAME = 'ActivityNet12'  # THUMOS14 / ActivityNet12

if config.DATASET_NAME == 'THUMOS14':
    config.DATASET = edict()
    config.DATASET.DATAROOT = os.getcwd() + '/data/' + config.DATASET_NAME
    config.DATASET.VIDEONUM_TRAIN = 200
    config.DATASET.VIDEONUM_TEST = 210
    config.DATASET.FEATURE_DIM = 1024
    config.DATASET.CLF_DIM = 21

    config.DATASET.SAMPLING_FRAMES = 25  # fps
    config.DATASET.NUM_INPUT_FRAMES = 16  # frames per segment
    config.DATASET.NUM_SEGMENTS = 400  # how many segments to sample

    config.DATASET.GT_JSON_PATH = os.path.join(config.DATASET.DATAROOT, 'test_data',
                                               'gt.json')
    config.DATASET.TEST_VID_LIST = 'THUMOS14_test_vid_list.txt'
    config.DATASET.TIOU_THRESHOLDS = np.linspace(0.1, 0.9, 9)


if config.DATASET_NAME == 'ActivityNet12':
    config.DATASET = edict()
    config.DATASET.DATAROOT = os.getcwd() + '/data/' + config.DATASET_NAME
    config.DATASET.VIDEONUM_TRAIN = 4819
    config.DATASET.VIDEONUM_TEST = 2383
    config.DATASET.FEATURE_DIM = 1024
    config.DATASET.CLF_DIM = 101

    config.DATASET.SAMPLING_FRAMES = 25
    config.DATASET.NUM_INPUT_FRAMES = 16
    config.DATASET.NUM_SEGMENTS = 400  # same as w-talc repo

    config.DATASET.GT_JSON_PATH = os.path.join(config.DATASET.DATAROOT, 'test_data',
                                               'gt_anet12.json')
    config.DATASET.TEST_VID_LIST = 'ActivityNet12_test_vid_list.txt'
    config.DATASET.TIOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)





# models related configs

config.MODEL = edict()

# config.MODEL.CONDITION_FRAME_NUM = 0



# training related configs

config.TRAIN = edict()

config.TRAIN.RESUME_RGB = False  # whether to continue previous training
config.TRAIN.RESUME_FLOW = False
config.TRAIN.STATE_DICT_RGB = 'train/baseline_rgb_0.1365483161624871.pth'
config.TRAIN.STATE_DICT_FLOW = 'train/checkpoint_flow_0.24359359838273537.pth'

config.TRAIN.OPTIMIZER = 'adam'  # sgd / adam
config.TRAIN.LR = 0.001
config.TRAIN.LR_DECAY_RATE = 0.5
config.TRAIN.LR_MILESTONES = []  # at which epoch lr decays
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 6000
config.TRAIN.BATCH_SIZE = 32

config.TRAIN.PRINT_EVERY_STEP = 1
config.TRAIN.TEST_EVERY_EPOCH = 10



# testing related configs

config.TEST = edict()
config.TEST.RESUME = True
config.TEST.STATE_DICT_RGB = 'train/final_rgb_0.27749696356005515.pth'
config.TEST.STATE_DICT_FLOW = 'train/final_flow_0.27749696356005515.pth'
config.TEST.BATCH_SIZE = 1
config.TEST.PRINT_EVERY_STEP = 1

config.TEST.SCALE = 20  # how many times to expand wTCAM by interpolating
config.TEST.CLASS_THRESHOLD = 0.1  # collect proposals of any class whose prob is over this thr

if config.DATASET_NAME == 'THUMOS14':
    config.TEST.FRAME_SCORE_RATIO_RGB = 0.35
    config.TEST.FRAME_SCORE_RATIO_FLOW = 0.65
    config.TEST.TEMPSEG_LIST_THR_RGB = 0.03
    config.TEST.TEMPSEG_LIST_THR_FLOW = 0.08
elif config.DATASET_NAME == 'ActivityNet12':
    config.TEST.FRAME_SCORE_RATIO_RGB = 0.5
    config.TEST.FRAME_SCORE_RATIO_FLOW = 0.5
    config.TEST.TEMPSEG_LIST_THR_RGB = 0.01
    config.TEST.TEMPSEG_LIST_THR_FLOW = 0.01




# extra settings

def extra():
    config.OUTPUT_DIR = os.path.join('experiments/', config.DATASET_NAME, config.MODE)
    config.TRAIN.STATE_DICT_RGB = os.path.join('experiments/', config.DATASET_NAME, config.TRAIN.STATE_DICT_RGB)
    config.TRAIN.STATE_DICT_FLOW = os.path.join('experiments/', config.DATASET_NAME, config.TRAIN.STATE_DICT_FLOW)
    config.TEST.STATE_DICT_RGB = os.path.join('experiments/', config.DATASET_NAME, config.TEST.STATE_DICT_RGB)
    config.TEST.STATE_DICT_FLOW = os.path.join('experiments/', config.DATASET_NAME, config.TEST.STATE_DICT_FLOW)




if config.DATASET_NAME == 'THUMOS14':
    config.DATASET.CLASS = ['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                            'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
                            'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
                            'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                            'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']
    config.DATASET.CLASS_INDEX = {'BaseballPitch': '7', 'BasketballDunk': '9', 'Billiards': '12',
                                  'CleanAndJerk': '21', 'CliffDiving': '22', 'CricketBowling': '23',
                                  'CricketShot': '24', 'Diving': '26', 'FrisbeeCatch': '31',
                                  'GolfSwing': '33', 'HammerThrow': '36', 'HighJump': '40',
                                  'JavelinThrow': '45', 'LongJump': '51', 'PoleVault': '68',
                                  'Shotput': '79', 'SoccerPenalty': '85', 'TennisSwing': '92',
                                  'ThrowDiscus': '93', 'VolleyballSpiking': '97'}
elif config.DATASET_NAME == 'ActivityNet12':
    config.DATASET.CLASS = ['Archery', 'Ballet', 'Bathing dog', 'Belly dance',
                            'Breakdancing', 'Brushing hair', 'Brushing teeth',
                            'Bungee jumping', 'Cheerleading', 'Chopping wood',
                            'Clean and jerk', 'Cleaning shoes', 'Cleaning windows',
                            'Cricket', 'Cumbia', 'Discus throw', 'Dodgeball',
                            'Doing karate', 'Doing kickboxing', 'Doing motocross',
                            'Doing nails', 'Doing step aerobics', 'Drinking beer',
                            'Drinking coffee', 'Fixing bicycle', 'Getting a haircut',
                            'Getting a piercing', 'Getting a tattoo', 'Grooming horse',
                            'Hammer throw', 'Hand washing clothes', 'High jump',
                            'Hopscotch', 'Horseback riding', 'Ironing clothes',
                            'Javelin throw', 'Kayaking', 'Layup drill in basketball',
                            'Long jump', 'Making a sandwich', 'Mixing drinks',
                            'Mowing the lawn', 'Paintball', 'Painting', 'Ping-pong',
                            'Plataform diving', 'Playing accordion', 'Playing badminton',
                            'Playing bagpipes', 'Playing field hockey', 'Playing flauta',
                            'Playing guitarra', 'Playing harmonica', 'Playing kickball',
                            'Playing lacrosse', 'Playing piano', 'Playing polo',
                            'Playing racquetball', 'Playing saxophone', 'Playing squash',
                            'Playing violin', 'Playing volleyball', 'Playing water polo',
                            'Pole vault', 'Polishing forniture', 'Polishing shoes',
                            'Preparing pasta', 'Preparing salad', 'Putting on makeup',
                            'Removing curlers', 'Rock climbing', 'Sailing', 'Shaving',
                            'Shaving legs', 'Shot put', 'Shoveling snow', 'Skateboarding',
                            'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Spinning',
                            'Springboard diving', 'Starting a campfire', 'Tai chi',
                            'Tango', 'Tennis serve with ball bouncing', 'Triple jump',
                            'Tumbling', 'Using parallel bars', 'Using the balance beam',
                            'Using the pommel horse', 'Using uneven bars',
                            'Vacuuming floor', 'Walking the dog', 'Washing dishes',
                            'Washing face', 'Washing hands', 'Windsurfing',
                            'Wrapping presents', 'Zumba']
    config.DATASET.CLASS_INDEX = {}

