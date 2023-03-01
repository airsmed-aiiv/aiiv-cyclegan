#!/usr/bin/python3

''' Projects based on https://github.com/aitorzip/PyTorch-CycleGAN.'''

import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % 3

import torch

from trainer import Trainer


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=10,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--cuda', default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
    opt = parser.parse_args()
    print(opt)
    return opt


def main():
    opt = parse_arg()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    trainer = Trainer(opt)
    trainer.train()

if __name__ == '__main__':
    main()