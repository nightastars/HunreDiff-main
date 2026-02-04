import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import argparse
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Testing Hybrid-domain Collaborative Universal Recursive Diffusion Model for Non-ideal Measurement CT Reconstruction under Extreme Degradation Conditions')
    parser.add_argument("--config", default='CT.yml', type=str,
                        help="Path to the config file")
    parser.add_argument('--resume', default='./save1-sino-stage2/ckpts/CT_69_ddpm.pth.tar', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=5,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['CT']")
    parser.add_argument("--save_path", default=r'./test', type=str,
                        help="Location to save restored images")
    parser.add_argument('--result_fig', type=bool, default=True,
                        help="Whether to save the NPY images as pictures PNG")
    parser.add_argument('--norm_range_min', type=float, default=-2.60,  # -1024.0
                        help="Maximum value of the image")
    parser.add_argument('--norm_range_max', type=float, default=180.0,  # -2048.0
                        help="Minimum value of the image")
    parser.add_argument('--trunc_min', type=float, default=-2.60,  # -1024.0
                        help="Maximum value of image truncation")
    parser.add_argument('--trunc_max', type=float, default=180.0,  # -2048.0
                        help="Minimum value for image truncation")
    parser.add_argument('--seed', default=None, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('--parse_patches', default=False, type=bool,
                        help='Whether Patch-Based testing')
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))
    fig_path = os.path.join(args.save_path, 'fig')
    x_path = os.path.join(args.save_path, 'x')
    y_path = os.path.join(args.save_path, 'y')
    pred_path = os.path.join(args.save_path, 'pred')
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print('Create path : {}'.format(fig_path))
    if not os.path.exists(x_path):
        os.makedirs(x_path)
        print('Create path : {}'.format(x_path))
    if not os.path.exists(y_path):
        os.makedirs(y_path)
        print('Create path : {}'.format(y_path))
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
        print('Create path : {}'.format(pred_path))
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
        print('Create path : {}'.format(fig_path))

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=args.parse_patches)

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    model.restore(val_loader, r=args.grid_r)


if __name__ == '__main__':
    main()
