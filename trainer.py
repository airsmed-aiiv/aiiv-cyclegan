'''
File: trainer.py
Project: PyTorch-CycleGAN
File Created: 2023-02-25 15:51:26
Author: sangminlee
-----
This script ...
Reference
...
'''
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from models import Generator
from models import Discriminator
from utils import weights_init_normal

from utils import LambdaLR
import itertools

from utils import ReplayBuffer
from utils import Logger

from us_datasets import USDataset
from torch.utils.data import DataLoader
from PIL import Image


class Trainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.load_networks()
        self.load_learning_config()
        self.load_data_config()

    def load_networks(self):
        self.netG_A2B = Generator(self.opt.input_nc, self.opt.output_nc)
        self.netG_B2A = Generator(self.opt.output_nc, self.opt.input_nc)
        self.netD_A = Discriminator(self.opt.input_nc)
        self.netD_B = Discriminator(self.opt.output_nc)

        if self.opt.cuda:
            self.netG_A2B.cuda()
            self.netG_B2A.cuda()
            self.netD_A.cuda()
            self.netD_B.cuda()

        self.netG_A2B.load_state_dict(torch.load('output_gelpad/netG_A2B_10.pth'))
        self.netG_B2A.load_state_dict(torch.load('output_gelpad/netG_B2A_10.pth'))
        self.netD_A.load_state_dict(torch.load('output_gelpad/netD_A_10.pth'))
        self.netD_B.load_state_dict(torch.load('output_gelpad/netD_B_10.pth'))

        self.netG_A2B.apply(weights_init_normal)
        self.netG_B2A.apply(weights_init_normal)
        self.netD_A.apply(weights_init_normal)
        self.netD_B.apply(weights_init_normal)

    def load_learning_config(self):
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                            lr=self.opt.lr, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.opt.lr, betas=(0.5, 0.999))

        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G,
                                                                lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch,
                                                                                   self.opt.decay_epoch).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A,
                                                                  lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch,
                                                                                     self.opt.decay_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B,
                                                                  lr_lambda=LambdaLR(self.opt.n_epochs, self.opt.epoch,
                                                                                     self.opt.decay_epoch).step)

    def load_data_config(self):
        # Inputs & targets memory allocation
        Tensor = torch.cuda.FloatTensor if self.opt.cuda else torch.Tensor

        self.input_A = Tensor(self.opt.batchSize, self.opt.input_nc, self.opt.size, self.opt.size)
        self.input_B = Tensor(self.opt.batchSize, self.opt.output_nc, self.opt.size, self.opt.size)
        self.target_real = Variable(Tensor(self.opt.batchSize).fill_(1.0), requires_grad=False)
        self.target_fake = Variable(Tensor(self.opt.batchSize).fill_(0.0), requires_grad=False)

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        # Dataset loader
        transforms_ = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        # transforms_ = [transforms.ToTensor()]
        ds = USDataset(transforms_=transforms_, unaligned=True)
        self.dataloader = DataLoader(ds, batch_size=self.opt.batchSize, shuffle=True, num_workers=self.opt.n_cpu)

        self.logger = Logger(self.opt.n_epochs, len(self.dataloader))

    def train(self):
        for epoch in range(self.opt.epoch, self.opt.n_epochs):
            for i, batch in enumerate(self.dataloader):
                if i == 2000:
                    break
                # Set model input
                real_A = Variable(self.input_A.copy_(batch['A']))
                real_B = Variable(self.input_B.copy_(batch['B']))
                ###### Generators A2B and B2A ######
                self.optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = self.netG_A2B(real_B)
                loss_identity_B = self.criterion_identity(same_B, real_B) * 10.0
                # G_B2A(A) should equal A if real A is fed
                same_A = self.netG_B2A(real_A)
                loss_identity_A = self.criterion_identity(same_A, real_A) * 5.0

                # GAN loss
                fake_B = self.netG_A2B(real_A)
                pred_fake = self.netD_B(fake_B)
                loss_GAN_A2B = 2. * self.criterion_GAN(pred_fake, self.target_real)

                fake_A = self.netG_B2A(real_B)
                pred_fake = self.netD_A(fake_A)
                loss_GAN_B2A = 1. * self.criterion_GAN(pred_fake, self.target_real)

                # Cycle loss
                recovered_A = self.netG_B2A(fake_B)
                loss_cycle_ABA = self.criterion_cycle(recovered_A, real_A) * 20.0

                recovered_B = self.netG_A2B(fake_A)
                loss_cycle_BAB = self.criterion_cycle(recovered_B, real_B) * 20.0

                # Total loss
                loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()

                self.optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                self.optimizer_D_A.zero_grad()

                # Real loss
                pred_real = self.netD_A(real_A)
                loss_D_real = self.criterion_GAN(pred_real, self.target_real)

                # Fake loss
                fake_A = self.fake_A_buffer.push_and_pop(fake_A)
                pred_fake = self.netD_A(fake_A.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) * 0.5
                loss_D_A.backward()

                self.optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                pred_real = self.netD_B(real_B)
                loss_D_real = self.criterion_GAN(pred_real, self.target_real)

                # Fake loss
                fake_B = self.fake_B_buffer.push_and_pop(fake_B)
                pred_fake = self.netD_B(fake_B.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, self.target_fake)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) * 1
                loss_D_B.backward()

                self.optimizer_D_B.step()
                ###################################

                self.logger.log({'loss_G': loss_G.cpu(), 'loss_G_identity': (loss_identity_A + loss_identity_B).cpu(),
                                 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).cpu(),
                                 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).cpu(),
                                 'loss_D': (loss_D_A + loss_D_B).cpu()},
                                images={'real_A': real_A.cpu(), 'real_B': real_B.cpu(), 'fake_A': fake_A.cpu(),
                                        'fake_B': fake_B.cpu()})

            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()

            # Save models checkpoints
            torch.save(self.netG_A2B.state_dict(), 'output_gelpad/netG_A2B_%d.pth' % (epoch + 10))
            torch.save(self.netG_B2A.state_dict(), 'output_gelpad/netG_B2A_%d.pth' % (epoch + 10))
            torch.save(self.netD_A.state_dict(), 'output_gelpad/netD_A_%d.pth' % (epoch + 10))
            torch.save(self.netD_B.state_dict(), 'output_gelpad/netD_B_%d.pth' % (epoch + 10))
        ###################################
