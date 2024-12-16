import logging
from .unet import Unet
from util.tools import loginfo


def build_Unet(args):
    return Unet(in_chans=args.in_chans, out_chans=args.out_chans, chans=args.chans,
                num_pool_layers=args.num_pool_layers, drop_prob=args.drop_prob)

model_factory = {
    'unet': build_Unet,
}

def build_model(args):
    assert args.model_name in model_factory.keys(), logging.error('unknown model name, please check')
    loginfo('crate model: {}'.format(args.model_name))
    return model_factory[args.model_name](args)
