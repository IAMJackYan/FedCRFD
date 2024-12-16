import argparse

modal_dict = {
    'A': 't1',
    'B': 't2',
    'C': 'ce',
}

def option():
    parser = argparse.ArgumentParser(description="xxx")
    parser.add_argument("--local_rank", type=int)

    # distributed setting
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)

    # dataset setting
    parser.add_argument('--dataset', default='brats', type=str)
    parser.add_argument('--root', default='xxx/data/fastMRI_brain_pkl_threemodals/', type=str)
    parser.add_argument('--mask_type', default=['random', 'equispaced', ''], type=str)
    parser.add_argument('--center_fractions', default=[0.08], type=float)
    parser.add_argument('--accelerations', default=[8, 4], type=int)
    parser.add_argument('--train_modals', default=['A', 'B', 'C'], type=list)
    parser.add_argument('--test_modals', default=['A', 'B', 'C'], type=list)
    parser.add_argument('--tgt_idx', default=0, type=int)
    parser.add_argument('--sample_rate', default=1.0, type=float)
    parser.add_argument('--merge_dataset', action='store_true', default=False)
    parser.add_argument('--vertical_rate', default=1.0, type=float)
    parser.add_argument('--pretrain_batch_size', default=8, type=int)
    parser.add_argument('--use_noise', action='store_true', default=False)
    parser.add_argument('--noise_rate', default=0.15, type=float)
    parser.add_argument('--use_gama', action='store_true', default=False)
    parser.add_argument('--gama', default=3.0, type=float)
    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--lam', default=0.2, type=float)
    parser.add_argument('--maskfiles', default=["/disk/yanyunlu/data/mask/1D-Uniform-5X_320.mat",
                                                "/disk/yanyunlu/data/mask/2D-Random-3X_320.mat",
                                                "/disk/yanyunlu/data/mask/1D-Cartesian_4X_320.mat"])

    # model setting
    parser.add_argument('--model_name', default='unet', type=str)
    parser.add_argument('--in_chans', default=1, type=int)
    parser.add_argument('--out_chans', default=1, type=int)
    parser.add_argument('--fl_method', default='tiantan', type=str)
    parser.add_argument('--u1', default=1.0, type=float)
    parser.add_argument('--u2', default=1.0, type=float)
    parser.add_argument('--u3', default=1.0, type=float)

    # model specific setting

    #unet
    parser.add_argument('--chans', default=32, type=int)
    parser.add_argument('--num_pool_layers', default=4, type=int)
    parser.add_argument('--drop_prob', default=0.0, type=float)

    # solver setting
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--comm_round', default=50, type=int)
    parser.add_argument('--print_freq', default=10, type=int)


    # save setting
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--work_dir', default='checkpoints', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--load', default='', type=str)
    parser.add_argument('--pretrain_epochs', default=50, type=int)

    args = parser.parse_args()

    args.train_modals = [modal_dict[c] for c in args.train_modals]
    args.test_modals = [modal_dict[c] for c in args.test_modals]

    return args
