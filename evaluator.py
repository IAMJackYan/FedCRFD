import time
import torch
import datetime

from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from util.metric import nmse, psnr, ssim, AverageMeter
from util.tools import loginfo
import matplotlib.pyplot as plt
import os

def vis_img(img, fname, ftype ,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(img, cmap='gray')
    figname = fname + '_' + ftype + '.png'
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)

class LocalEvaluator(object):
    def __init__(self, args, dataset, device, model):
        self.dataset = dataset
        self.args = args
        self.device = device
        self.model = model

        sampler = DistributedSampler(dataset, shuffle=False) if args.distributed else SequentialSampler(dataset)
        self.dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                    sampler=sampler, num_workers=args.num_worker,
                                    pin_memory=True)
        self.dataset_len = len(dataset)
        self.criterion = torch.nn.L1Loss().to(device)

    @torch.no_grad()
    def single_gpu_evaluate(self):
        self.model.eval()
        nmse_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        loss_meter = AverageMeter()
        for idx, data in enumerate(tqdm(self.dataloader)):
            x = data[0]
            target = data[1]

            mean = data[2]
            std = data[3]

            mean = mean.float()
            std = std.float()

            mean = mean.unsqueeze(1).unsqueeze(2).to(self.device)
            std = std.unsqueeze(1).unsqueeze(2).to(self.device)

            b = x.shape[0]

            x = x.float().to(self.device)
            target = target.float().to(self.device)
            output = self.model(x)

            loss = self.criterion(output, target)
            loss_meter.update(loss, 1)

            output = output * std + mean
            target = target * std + mean
            # image = x * std + mean

            for i in range(b):
                nmse_v = nmse(target[i].cpu().numpy(), output[i].cpu().numpy())
                psnr_v = psnr(target[i].cpu().numpy(), output[i].cpu().numpy())
                ssim_v = ssim(target[i].cpu().numpy(), output[i].cpu().numpy())

                nmse_meter.update(nmse_v, 1)
                psnr_meter.update(psnr_v, 1)
                ssim_meter.update(ssim_v, 1)

        return nmse_meter.avg, psnr_meter.avg, ssim_meter.avg, loss_meter.avg

    def distributed_concat(self, tensor, num_total_examples):
        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)
        concat = torch.cat(output_tensors, dim=0)
        # truncate the dummy elements added by SequentialDistributedSampler
        return concat[:num_total_examples]

    @torch.no_grad()
    def multi_gpu_eval_evaluate(self):
        self.model.eval()

        nmse_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        output_list = []
        target_list = []

        for data in tqdm(self.dataloader):
            x = data[0]
            target = data[1]

            mean = data[2]
            std = data[3]

            mean = mean.unsqueeze(1).unsqueeze(2).to(self.device)
            std = std.unsqueeze(1).unsqueeze(2).to(self.device)

            x = x.float().to(self.device)
            target = target.float().to(self.device)
            output = self.model(x)

            output = output * std + mean
            target = target * std + mean

            output_list.append(output)
            target_list.append(target)

        final_output = self.distributed_concat(torch.cat((output_list), dim=0), self.dataset_len)
        final_target = self.distributed_concat(torch.cat((target_list), dim=0), self.dataset_len)

        b = final_output.shape[0]

        for i in range(b):
            nmse_v = nmse(final_target[i].cpu().numpy(), final_output[i].cpu().numpy())
            psnr_v = psnr(final_target[i].cpu().numpy(), final_output[i].cpu().numpy())
            ssim_v = ssim(final_target[i].cpu().numpy(), final_output[i].cpu().numpy())

            nmse_meter.update(nmse_v, 1)
            psnr_meter.update(psnr_v, 1)
            ssim_meter.update(ssim_v, 1)

        loss = self.criterion(final_output, final_target)
        return nmse_meter.avg, psnr_meter.avg, ssim_meter.avg, loss.detach().item()

    def evaluate(self, round, idx):
        loginfo('>>>> Round:{} client: {} evaluating >>>>'.format(round, idx))
        start_time = time.time()
        if self.args.distributed:
            nmse_avg, psnr_avg, ssim_avg, loss = self.multi_gpu_eval_evaluate()
        else:
            nmse_avg, psnr_avg, ssim_avg, loss = self.single_gpu_evaluate()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        loginfo("NMSE: {:.4}".format(nmse_avg))
        loginfo("PSNR: {:.4}".format(psnr_avg))
        loginfo("SSIM: {:.4}".format(ssim_avg))
        loginfo("validation loss:{:.4}".format(loss))
        loginfo('Evaluate time {}'.format(total_time_str))
        loginfo('>>>>>>>>>>>complete>>>>>>>>>>>>>>')

        return {'NMSE': nmse_avg, 'PSNR': psnr_avg, 'SSIM': ssim_avg}

