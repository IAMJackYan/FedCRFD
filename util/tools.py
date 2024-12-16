import logging
import os
import pickle
import numpy as np

from pathlib import Path
from util.misc import is_main_process

def set_for_logger(args):

    # log_filename = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '.txt'

    log_filename = str(os.getpid())
    log_filepath = os.path.join(args.log_dir, log_filename)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def loginfo(str):
    if is_main_process():
        logging.info(str)


def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def fft2(img):
    return np.fft.fftshift(np.fft.fft2(img))

def average_eval(eval_list):
    eval_result = {'NMSE': 0, 'PSNR': 0, 'SSIM': 0}
    for status in eval_list:
        eval_result['NMSE'] += status['NMSE']
        eval_result['PSNR'] += status['PSNR']
        eval_result['SSIM'] += status['SSIM']
    eval_result['NMSE'] /= len(eval_list)
    eval_result['SSIM'] /= len(eval_list)
    eval_result['PSNR'] /= len(eval_list)
    return eval_result