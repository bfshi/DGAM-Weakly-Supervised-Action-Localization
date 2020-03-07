# This code is originally from the official ActivityNet repo
# https://github.com/activitynet/ActivityNet
# Small modification from ActivityNet Code

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from eval_detection import ANETdetection
import _init_paths
from lib.core.config import config

tiou_thre_dict = {'THUMOS14': np.linspace(0.1, 0.9, 9),
                   'ActivityNet12': np.linspace(0.5, 0.95, 10)}


def main(ground_truth_filename, prediction_filename,
         subset='test', tiou_thresholds='THUMOS14',
         verbose=True, check_status=True):

    tiou_thresholds = tiou_thre_dict[tiou_thresholds]

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=(config.DATASET_NAME == 'ActivityNet12'))
    anet_detection.evaluate()

def eval_mAP(ground_truth_filename, prediction_filename,
         subset='test', tiou_thresholds=np.linspace(0.1, 0.9, 9),
         verbose=True, check_status=True):
    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=(config.DATASET_NAME == 'ActivityNet12'))
    return anet_detection.evaluate()

def parse_input():
    description = ('This script allows you to evaluate the ActivityNet '
                   'detection task which is intended to evaluate the ability '
                   'of  algorithms to temporally localize activities in '
                   'untrimmed video sequences.')
    p = argparse.ArgumentParser(description=description)
    p.add_argument('ground_truth_filename',
                   help='Full path to json file containing the ground truth.')
    p.add_argument('prediction_filename',
                   help='Full path to json file containing the predictions.')
    p.add_argument('--subset', default='test',
                   help=('String indicating subset to evaluate: '
                         '(training, validation)'))
    p.add_argument('--tiou_thresholds', type=str, default='THUMOS14',
                   help='Temporal intersection over union threshold.')
    p.add_argument('--verbose', type=bool, default=True)
    p.add_argument('--check_status', type=bool, default=True)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_input()
    main(**vars(args))
