import copy
import random
import numpy as np
import os

from pathlib import Path
from util.misc import init_distributed_mode, is_main_process, get_rank, save_checkpoint
from util.opt import option
from data import load_fastMRI
from util.tools import set_for_logger, loginfo
from trainer import VerticalTrainer, HorizontalTrainer
from evaluator import LocalEvaluator
from models.unet import *
from util.tools import average_eval
from torch.autograd import Variable


def communication(server_model, models, client_weights):
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key])
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

def init(models, global_model):
    with torch.no_grad():
        for model in models:
            for key in model.state_dict().keys():
                if 'encoder' in key:
                    continue
                else:
                    model.state_dict()[key].data.copy_(global_model.state_dict()[key])

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

    global_model = Unet(in_chans=args.in_chans, out_chans=args.out_chans, chans=args.chans,
                num_pool_layers=args.num_pool_layers, drop_prob=args.drop_prob)
    global_model = global_model.to(device)

    local_models = [UnetModel_ad_da(in_chans=args.in_chans, out_chans=args.out_chans, chans=args.chans,
                num_pool_layers=args.num_pool_layers, drop_prob=args.drop_prob).to(device) for _ in range(client_nums)]

    central_classifier = Classifier(num_classes=client_nums).to(device)

    trainset_dict = load_fastMRI(args, args.train_modals, is_train=True, use_gama=args.use_gama, gama=args.gama)
    valset_dict = load_fastMRI(args, args.test_modals, is_train=False)

    client_weight = [ 1 / client_nums for _ in range(client_nums)]


    best_status = [None for _ in range(client_nums)]

    if os.path.exists(args.resume):
        checkpoint = torch.load(args.resume)
        global_model = global_model.load_state_dict(checkpoint['model'], strict=True)
        start_round = checkpoint['round'] + 1
        best_status = checkpoint['best_status']
        loginfo('resume checkpoint from {}'.format(args.resume))

    if args.distributed:
        global_model = torch.nn.parallel.DistributedDataParallel(global_model, device_ids=[args.gpu])
        central_classifier = torch.nn.parallel.DistributedDataParallel(central_classifier, device_ids=[args.gpu])
        for i in range(len(local_models)):
            local_models [i] = torch.nn.parallel.DistributedDataParallel(local_models[i], device_ids=[args.gpu])

    # save model
    work_path = os.path.join(args.work_dir, args.fl_method, str(os.getpid()))
    loginfo('the checkpoint will be save at {}'.format(work_path))
    Path(work_path).mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = os.path.join(work_path, 'last.pth')

    labels = {}
    for i in range(client_nums):
        t = np.zeros((args.batch_size, client_nums))
        t[..., i] = 1
        t = Variable(torch.LongTensor(t).to(device), requires_grad=False)
        labels[i] = t

    best_status_epoch = None

    init(local_models, global_model)


    for round in range(start_round, args.comm_round+1):
        hTrainer = HorizontalTrainer(args=args, dataset_dict=trainset_dict, device=device, models=local_models,
                                      classifier=central_classifier, labels=labels)
        vTrainer = VerticalTrainer(args=args, dataset_dict=trainset_dict, device=device, models=local_models,
                                   classifier=central_classifier, labels=labels)
        for epoch in range(1, args.epochs+1):
            _ = hTrainer.train(round, epoch)
            w_locals = vTrainer.train(round, epoch)

        global_model, local_models = communication(global_model, local_models, client_weight)

        eval_list = []
        for idx, dataset in valset_dict.items():
            local_evaluator = LocalEvaluator(args=args, dataset=dataset, device=device, model=local_models[idx])
            eval_status = local_evaluator.evaluate(round, idx)
            eval_list.append(eval_status)
            if best_status[idx] is None or best_status[idx]['PSNR'] < eval_status['PSNR']:
                best_status[idx] = eval_status
                best_status[idx]['round'] = round
                best_checkpoint_path = os.path.join(work_path, '{}_best.pth'.format(args.test_modals[idx]))
                save_checkpoint({
                    'model': local_models[idx].module.state_dict() if args.distributed else local_models[idx].state_dict(),
                    'round': round,
                    'args': args,
                }, best_checkpoint_path)

        average_eval_status = average_eval(eval_list)

        for idx, model in enumerate(local_models):
            save_checkpoint({
                'model': model.module.state_dict() if args.distributed else model.state_dict(),
                'round': round,
                'args': args,
                'best_status': best_status
            }, os.path.join(work_path, '{}_last.pth'.format(args.test_modals[idx])))

        if best_status_epoch is None or average_eval_status['PSNR'] > best_status_epoch['PSNR']:
            best_status_epoch = average_eval_status
            best_status_epoch['round'] = round
            for idx, model in enumerate(local_models):
                save_checkpoint({
                'model': global_model.module.state_dict() if args.distributed else global_model.state_dict(),
                'round': round,
                'args': args,
            }, os.path.join(work_path, '{}_best_round.pth'.format(args.test_modals[idx])))


    loginfo('===========>>>>>Final result===========>>>>>')
    for idx, status in enumerate(best_status):
        loginfo('{}: Best round:{} NMSE({:.4}) PSNR({:.4}) SSIM({:.4})'.format(args.test_modals[idx], best_status[idx]['round'],
                                                                                    best_status[idx]['NMSE'],
                                                                                    best_status[idx]['PSNR'],
                                                                                    best_status[idx]['SSIM']))
    loginfo('Best round:{} NMSE({:.4}) PSNR({:.4}) SSIM({:.4}'.format(best_status_epoch['round'], best_status_epoch['NMSE'],
                                                                      best_status_epoch['PSNR'], best_status_epoch['SSIM']))
if __name__ == '__main__':
    args = option()
    main(args)