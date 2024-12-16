import os
import random

import numpy as np
import torch
import logging

from util.tools import pkload, fft2
from torch.utils.data import Dataset
from .transforms import *


def augment_gamma(data_sample, gamma_range=(0.5, 2), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats: Union[bool, Callable[[], bool]] = False):
    if invert_image:
        data_sample = - data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = - data_sample
    return data_sample

class FastMRI(Dataset):
    def __init__(self, list_file, root='', modal='t1', ids=None, mask=None, sample_rate = 1.0, use_gama = False):
        paths = []
        with open(list_file) as f:
            lines = f.readlines()
            dataset_len = int(len(lines) * sample_rate)
            lines = lines[:dataset_len]
            if ids is not None:
                assert max(ids)< len(lines) and min(ids) >=0, logging.error('error sample id, please check')
                lines = [lines[id] for id in ids]
            for line in lines:
                line = line.strip()
                name = line.split('.')[0]
                path = os.path.join(root, line)
                slices = pkload(path)['slice']
                for slice in range(slices):
                    paths.append((path, slice, name))

        self.examples = paths
        self.mask = mask
        self.modal = modal
        self.use_gama = use_gama
    def __getitem__(self, item):

        path, slice, name = self.examples[item]
        data = pkload(path)
        image = data[self.modal]
        image = image[slice]

        p = random.uniform(0, 1)
        if self.use_gama and p > 0.5:
            image = augment_gamma(image)

        target = image
        target = to_tensor(target)
        kspace = fft2(image)
        mask_kspace = kspace * self.mask
        image = np.fft.ifft2(mask_kspace)
        image = abs(image)
        image = to_tensor(image)

        image, mean, std = normalize_instance(image, eps=1e-11)
        target = normalize(target, mean, std, eps=1e-11)

        return image, target, mean, std, name, slice

    def __len__(self):
        return len(self.examples)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]


class Merge(Dataset):
    def __init__(self, list_file, root='', modal=['t1'], ids=None, mask=None, sample_rate = 1.0):
        paths = []
        with open(list_file) as f:
            lines = f.readlines()
            dataset_len = int(len(lines) * sample_rate)
            lines = lines[:dataset_len]
            if ids is not None:
                assert max(ids)< len(lines) and min(ids) >=0, logging.error('error sample id, please check')
                lines = [lines[id] for id in ids]
            for line in lines:
                line = line.strip()
                name = line.split('.')[0]
                path = os.path.join(root, line)
                for slice in range(16):
                    for m in modal:
                        paths.append((path, slice, name, m))

        self.examples = paths
        self.mask = mask
        self.modal = modal
    def __getitem__(self, item):

        path, slice, name = self.examples[item]
        data = pkload(path)
        image = data[self.modal]
        image = image[slice]

        target = image
        kspace = fft2(image)
        mask_kspace = kspace * self.mask
        image = np.fft.ifft2(mask_kspace)
        image = abs(image)
        image = to_tensor(image)

        image, mean, std = normalize_instance(image, eps=1e-11)
        target = normalize(target, mean, std, eps=1e-11)

    def __len__(self):
        return len(self.examples)

    def collate(self, batch):
        return [torch.cat(v) for v in zip(*batch)]



