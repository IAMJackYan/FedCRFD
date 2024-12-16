import copy
import torch
import random
import numpy as np
import os
import time
import datetime
from pathlib import Path
from util.misc import init_distributed_mode, is_main_process, get_rank, save_checkpoint
from util.opt import option
from data import load_fastMRI
from util.tools import set_for_logger, loginfo
from models.unet import *
from trainer import LocalTrainer
from evaluator import LocalEvaluator
from util.tools import average_eval
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from util.metric import nmse, psnr, ssim, AverageMeter
from collections import defaultdict
from tqdm import tqdm
import h5py
def save_reconstructions(reconstructions, out_dir):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input
            filenames to corresponding reconstructions (of shape num_slices x
            height x width).
        out_dir (pathlib.Path): Path to the output directory where the
            reconstructions should be saved.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    print(out_dir,len(list(reconstructions.keys())))
    #idx = min(len(list(reconstructions.keys())), 10)
    for fname in list(reconstructions.keys()):
        f_output = torch.stack([v for _, v in reconstructions[fname].items()])

        basename = fname.split('/')[-1]
        with h5py.File(str(out_dir) + '/' + str(basename) + '.hdf5', "w") as f:
            print(fname)
            f.create_dataset("reconstruction", data=f_output.cpu())


@torch.no_grad()
def visual_output(model, data_loader, device, output_dir):
    model.eval()
    nmse_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    output_dic = defaultdict(dict)
    target_dic = defaultdict(dict)
    input_dic = defaultdict(dict)

    start_time = time.time()

    for data in tqdm(data_loader):
        x, target, mean, std, fname, slice_num = data  # NOTE

        mean = mean.float()
        std = std.float()

        mean = mean.unsqueeze(1).unsqueeze(2).to(device)
        std = std.unsqueeze(1).unsqueeze(2).to(device)

        b = x.shape[0]

        x = x.float().to(device)
        target = target.float().to(device)
        output = model(x)

        output = output * std + mean
        target = target * std + mean
        image = x * std + mean

        for i, f in enumerate(fname):
            output_dic[f][slice_num[i]] = output[i]
            target_dic[f][slice_num[i]] = target[i]
            input_dic[f][slice_num[i]] = image[i]

        for i in range(b):
            nmse_v = nmse(target[i].cpu().numpy(), output[i].cpu().numpy())
            psnr_v = psnr(target[i].cpu().numpy(), output[i].cpu().numpy())
            ssim_v = ssim(target[i].cpu().numpy(), output[i].cpu().numpy())

            nmse_meter.update(nmse_v, 1)
            psnr_meter.update(psnr_v, 1)
            ssim_meter.update(ssim_v, 1)

    save_reconstructions(output_dic, os.path.join(output_dir, 'output'))
    save_reconstructions(input_dic, os.path.join(output_dir, 'input'))
    save_reconstructions(target_dic, os.path.join(output_dir, 'target'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print(' Evaluate time {} NMSE: {:.4f} PSNR: {:.4f} SSIM: {:.4f}'.format(
                                                                        total_time_str,
                                                                        nmse_meter.avg,
                                                                        psnr_meter.avg,
                                                                        ssim_meter.avg))


def main(args):
    init_distributed_mode(args)

    client_nums = len(args.train_modals)

    if is_main_process():
        set_for_logger(args)

    loginfo(args)
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    start_round = 1
    device = torch.device(args.device)

    global_model = UnetModel_z(in_chans=args.in_chans, out_chans=args.out_chans, chans=args.chans,
                num_pool_layers=args.num_pool_layers, drop_prob=args.drop_prob).to(device)

    valset_dict = load_fastMRI(args, args.test_modals, is_train=False)


    if os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        global_model.load_state_dict(checkpoint['model'], strict=True)
        loginfo('resume checkpoint from {}'.format(args.resume))

    if args.distributed:
        global_model = torch.nn.parallel.DistributedDataParallel(global_model, device_ids=[args.gpu])


    for idx, dataset in valset_dict.items():
        sampler = DistributedSampler(dataset, shuffle=False) if args.distributed else SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                    sampler=sampler, num_workers=args.num_worker,
                                    pin_memory=True)
        output_path = os.path.join('tiantan_output2', 'solo3', args.test_modals[idx])
        visual_output(global_model, dataloader, device, output_dir=output_path)

if __name__ == '__main__':
    args = option()
    main(args)