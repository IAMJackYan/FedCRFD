import os
import numpy as np

from .datasets import  FastMRI
from .transforms import UnetDataTransform
from .subsample import create_mask_for_mask_type
from util.tools import loginfo
from scipy.io import loadmat

def build_transforms(mask_type, center_fractions, accelerations, is_train = True, use_noise = False, noise_rate = 0.15, use_gama=False, gama=3.0):

    mask = create_mask_for_mask_type(
        mask_type, center_fractions, accelerations
    )
    if is_train:
        return UnetDataTransform(mask_func=mask, use_seed=False, noise_aug=use_noise, noise_rate=noise_rate, use_gama=use_gama, gama=gama)
    else:
        return UnetDataTransform(mask_func=mask)

#partition dataset into each client
def parition_dataset(args):
    client_nums = len(args.train_modals)
    list_file = os.path.join(args.root, 'train', 'train.txt')
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ids = range(int(len(lines)* args.sample_rate))
    ids = np.random.permutation(ids)
    split = round(len(ids) * 0.1)
    vertical_space = ids[:split]
    horizontal_space = ids[split:]
    client_hids = np.array_split(horizontal_space, client_nums)
    vertical_length = round(args.vertical_rate * len(vertical_space))
    client_vids = vertical_space[:vertical_length]
    hid_map = {i:client_hids[i] for i in range(client_nums)}
    return hid_map, client_vids

def parition_fastmri_dataset(args):
    client_nums = len(args.train_modals)
    list_file = os.path.join(args.root, 'train.txt')
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ids = range(int(len(lines)* args.sample_rate))
    ids = np.random.permutation(ids)
    split = round(len(ids) * 0.1)
    vertical_space = ids[:split]
    horizontal_space = ids[split:]
    client_hids = np.array_split(horizontal_space, client_nums)
    vertical_length = round(args.vertical_rate * len(vertical_space))
    print(vertical_length)
    client_vids = vertical_space[:vertical_length]
    hid_map = {i:client_hids[i] for i in range(client_nums)}
    return hid_map, client_vids


def load_fastMRI(args, modals, is_train = True, use_noise = False, noise_rate = 0.15, use_gama=False, gama=3.0):
    dataset_dict = {}
    id_map, vids = parition_fastmri_dataset(args)
    for idx, modal in enumerate(modals):
        mask = loadmat(args.maskfiles[idx])['mask']
        if is_train:
            loginfo('------build train dataset------')
            root = args.root
            list_file = os.path.join(args.root, 'train.txt')
            if args.merge_dataset:
                ids = np.append(vids, id_map[idx])
                dataset = FastMRI(list_file=list_file, root=root, modal=modal, mask=mask, ids=ids, sample_rate=args.sample_rate)
                dataset_dict[idx] = dataset
                loginfo('{} : create merge dataset, length ---> {}'.format(modal, len(dataset)))
            else:
                vertical_dataest = FastMRI(list_file=list_file, root=root, modal=modal,  mask=mask, ids= vids, sample_rate=args.sample_rate, use_gama=use_gama)
                horizontal_dataset = FastMRI(list_file=list_file, root=root, modal=modal, mask=mask, ids= id_map[idx], sample_rate=args.sample_rate, use_gama=use_gama)
                dataset_dict[idx] = (vertical_dataest, horizontal_dataset)
                loginfo('{} : create vertical dataset, length ---> {}'.format(modal, len(vertical_dataest)))
                loginfo('{} : create horizontal dataset, length ---> {}'.format(modal,len(horizontal_dataset)))

        else:
            loginfo('------build val dataset------')
            root = args.root
            list_file = os.path.join(args.root, 'test.txt')
            dataset = FastMRI(list_file=list_file, root=root, modal=modal,  mask=mask, sample_rate=args.sample_rate)
            dataset_dict[idx] = dataset
            loginfo('{} : create val dataset, length ---> {}'.format(modal, len(dataset)))
    return dataset_dict
