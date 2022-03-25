"""
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse
import torch
from torch.backends import cudnn
from munch import Munch
from core.data_loader import get_train_loader, get_test_loader
from core.solver import Solver
from core.utils import get_config


def main(args):
    config = get_config(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda_device']
    cudnn.benchmark = True
    solver = Solver(config)
    if config['mode'] == 'train':
        loaders = Munch(src=get_train_loader(root=config['train_img_dir'],
                                             img_size=config['img_size'],
                                             batch_size=config['batch_size'],
                                             num_workers=config['num_workers']))
        solver.train(loaders)
    elif config['mode'] == 'test':
        loaders = Munch(src=get_test_loader(root=config['test_img_dir'],
                                            test_img_list=config['test_img_list'],
                                            img_size=config['img_size'],
                                            batch_size=config['batch_size'],
                                            num_workers=config['num_workers']))
        solver.test(loaders)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='param.yaml', help='Path to the config file.')
    args = parser.parse_args()
    main(args)
