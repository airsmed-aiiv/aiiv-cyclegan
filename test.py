#!/usr/bin/python3

import argparse
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % 3
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from PIL import Image

from models import Generator
from us_datasets import USDataset

from pathlib import Path
from utils import tensor2image
from tqdm import tqdm
import time

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=16, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/sangmin.lee/Dataset/US simulation & segmentation/cyclegan',
                    help='root directory of the dataset')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_gelpad = Generator(opt.input_nc, opt.output_nc)
# netG_alcohol = Generator(opt.output_nc, opt.input_nc)

netG_gelpad.cuda()
# netG_alcohol.cuda()

# Load state dicts
netG_gelpad.load_state_dict(torch.load('output_gelpad/netG_A2B_10.pth'))
# netG_alcohol.load_state_dict(torch.load('output_alcohol/netG_A2B_1.pth'))

# Set model's test mode
netG_gelpad.eval()
# netG_alcohol.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
transforms_ = [transforms.ToTensor()]
# alcohol_dl = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, mode='test', medium='alcohol'),
#                         batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
gelpad_dl = DataLoader(USDataset(opt.dataroot, transforms_=transforms_, mode='test', medium='gelpad'),
                       batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)
###################################

print('Here')

with torch.no_grad():
    # for i, batch in tqdm(enumerate(alcohol_dl)):
        # out = netG_alcohol(batch['A'].cuda())
        # for n in range(out.shape[0]):
        #     fn_target = batch['fn'][n].replace('RB', 'RB2AE2')
        #     Path(fn_target).parent.mkdir(parents=True, exist_ok=True)
        #     Image.fromarray(tensor2image(out[n].cpu().detach())).save(fn_target)
        # break
    for i, batch in tqdm(enumerate(gelpad_dl)):
        out = netG_gelpad(batch['A'].cuda())
        for n in range(out.shape[0]):
            fn_target = batch['fn'][n].replace('RB', 'RB2AE_gelpad')
            Path(fn_target).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(tensor2image(out[n].cpu().detach())).save(fn_target)
