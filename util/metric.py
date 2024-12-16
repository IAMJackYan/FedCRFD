
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def nmse(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """Compute Structural Similarity Index Metric (SSIM)"""
    return structural_similarity(gt, pred, data_range=gt.max())


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

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
        self.avg = self.sum / self.count