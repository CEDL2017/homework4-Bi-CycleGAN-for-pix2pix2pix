import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class BiCycleGANModel(BaseModel): # johnson
    def name(self):
        return 'BiCycleGANModel' # johnson

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        # johnson
        self.input_B = self.Tensor(nb, opt.intermediate_nc, size, size)
        self.input_C = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        # johnson
        self.netG_A = networks.define_G(opt.input_nc, opt.intermediate_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.intermediate_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_C = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # johnson
            self.netD_A = networks.define_D(opt.intermediate_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_C = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            # johnson
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            self.load_network(self.netG_C, 'G_C', which_epoch)
            if self.isTrain:
                # johnson
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_C, 'D_C', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_C_pool = ImagePool(opt.pool_size) # johnson
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), 
                                                                self.netG_B.parameters(), self.netG_C.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999)) # johnson
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) # johnson
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_C) # johnson
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        # johnson
        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)
        networks.print_network(self.netG_C)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_C)
        print('-----------------------------------------------')

    def set_input(self, input):
        ## johnson NOTE
        if self.opt.which_direction == 'AtoBtoC':
            input_A = input['A']
            input_B = input['B']
            input_C = input['C']
            self.image_paths = input['A_paths']
        elif self.opt.which_direction == 'BtoCtoA':
            input_A = input['B']
            input_B = input['C']
            input_C = input['A']
            self.image_paths = input['B_paths']
        elif self.opt.which_direction == 'CtoAtoB':
            input_A = input['C']
            input_B = input['A']
            input_C = input['B']
            self.image_paths = input['C_paths']
        else:
            raise ValueError("Invalid opt.which_direction {}".format(self.opt.which_direction))
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C) # johnson

    def test(self):
        ## johnson
        real_A = Variable(self.input_A, volatile=True)
        fake_B = self.netG_A(real_A)
        self.rec_A = self.netG_C(self.netG_B(fake_B)).data
        self.fake_B = fake_B.data

        real_B = Variable(self.input_B, volatile=True)
        fake_C = self.netG_B(real_B)
        self.rec_B = self.netG_A(self.netG_C(fake_C)).data
        self.fake_C = fake_C.data

        real_C = Variable(self.input_C, volatile=True)
        fake_A = self.netG_C(real_C)
        self.rec_C = self.netG_B(self.netG_A(fake_A)).data
        self.fake_A = fake_A.data

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # johnson
    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

    # johnson 
    def backward_D_B(self):
        fake_C = self.fake_C_pool.query(self.fake_C)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_C, fake_C)
        self.loss_D_B = loss_D_B.data[0]

    # johnson
    def backward_D_C(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_C = self.backward_D_basic(self.netD_C, self.real_A, fake_A)
        self.loss_D_C = loss_D_C.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C # johnson
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A = self.netG_A(self.real_B) # this should look like B
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_C is fed. johnson
            idt_B = self.netG_B(self.real_C) # this should look like C
            loss_idt_B = self.criterionIdt(idt_B, self.real_C) * lambda_C * lambda_idt
            # G_C should be identity if real_A is fed. johnson
            idt_C = self.netG_C(self.real_A) # this should look like A
            loss_idt_C = self.criterionIdt(idt_C, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.idt_C = idt_C.data # johnson
            self.loss_idt_A = loss_idt_A.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
            self.loss_idt_C = loss_idt_C.data[0] # johnson
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            loss_idt_C = 0 # johnson
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            self.loss_idt_C = 0 # johnson

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B)), johnson
        fake_C = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_C)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # GAN loss D_C(G_C(C)), johnson
        fake_A = self.netG_C(self.real_C)
        pred_fake = self.netD_C(fake_A)
        loss_G_C = self.criterionGAN(pred_fake, True)

        # A-->B cycle loss
        rec_A = self.netG_C(self.netG_B(fake_B)) # johnson
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # B-->C cycle loss
        rec_B = self.netG_A(self.netG_C(fake_C)) # johnson
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        # C-->A cycle loss, johnson
        rec_C = self.netG_B(self.netG_A(fake_A))
        loss_cycle_C = self.criterionCycle(rec_C, self.real_C) * lambda_C

        # combined loss
        loss_G = loss_G_A + loss_G_B + loss_G_C + \
                 loss_cycle_A + loss_cycle_B + loss_cycle_C + \
                 loss_idt_A + loss_idt_B + loss_idt_C # johnson
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.fake_C = fake_C.data # johnson
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data
        self.rec_C = rec_C.data # johnson

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_G_C = loss_G_C.data[0] # johnson
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        self.loss_cycle_C = loss_cycle_C.data[0] # johnson

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B and G_C (johnson)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # D_C johnson
        self.optimizer_D_C.zero_grad()
        self.backward_D_C()
        self.optimizer_D_C.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                  ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B', self.loss_cycle_B),
                                  ('D_C', self.loss_D_C), ('G_C', self.loss_G_C), ('Cyc_C', self.loss_cycle_C)]) # johnson
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
            ret_errors['idt_C'] = self.loss_idt_C # johnson
        return ret_errors

    def get_current_visuals(self):
        # A --> B
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)

        # B --> C, johnson
        real_B = util.tensor2im(self.input_B)
        fake_C = util.tensor2im(self.fake_C)
        rec_B = util.tensor2im(self.rec_B)

        # C --> A, johnson
        real_C = util.tensor2im(self.input_C)
        fake_A = util.tensor2im(self.fake_A)
        rec_C = util.tensor2im(self.rec_C)

        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_C', fake_C), ('rec_B', rec_B),
                                   ('real_C', real_C), ('fake_A', fake_A), ('rec_C', rec_C)]) # johnson
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
            ret_visuals['idt_C'] = util.tensor2im(self.idt_C) # johnson
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        # johnson 
        self.save_network(self.netD_C, 'D_C', label, self.gpu_ids)
        self.save_network(self.netG_C, 'G_C', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
