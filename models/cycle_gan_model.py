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


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_C = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_AB = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_AC = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_C = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_AB = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_AC = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_C = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch)
            self.load_network(self.netG_AC, 'G_AC', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            self.load_network(self.netG_C, 'G_C', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_AB, 'D_AB', which_epoch)
                self.load_network(self.netD_AC, 'D_AC', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_C, 'D_C', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_C_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_GB = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizer_GC = torch.optim.Adam(itertools.chain(self.netG_AC.parameters(), self.netG_C.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            
            self.optimizer_D_AB = torch.optim.Adam(self.netD_AB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_AC = torch.optim.Adam(self.netD_AC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_GB)
            self.optimizers.append(self.optimizer_GC)
            self.optimizers.append(self.optimizer_D_AB)
            self.optimizers.append(self.optimizer_D_AC)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_C)
            
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_AB)
        networks.print_network(self.netG_AC)
        networks.print_network(self.netG_B)
        networks.print_network(self.netG_C)
        if self.isTrain:
            networks.print_network(self.netD_AB)
            networks.print_network(self.netD_AC)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_C)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        input_C = input['C']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        real_B = Variable(self.input_B, volatile=True)
        real_C = Variable(self.input_C, volatile=True)

        fake_AB = self.netG_B(real_B)
        fake_AC = self.netG_C(real_C)
        fake_B = self.netG_AB(real_A)
        fake_C = self.netG_AC(real_A)

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

    def backward_D_AB(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_AB = self.backward_D_basic(self.netD_AB, self.real_B, fake_B)
        self.loss_D_AB = loss_D_AB.data[0]

    def backward_D_AC(self):
        fake_C = self.fake_C_pool.query(self.fake_C)
        loss_D_AC = self.backward_D_basic(self.netD_AC, self.real_C, fake_C)
        self.loss_D_AC = loss_D_AC.data[0]

    def backward_D_B(self):
        fake_AB = self.fake_A_pool.query(self.fake_AB)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_AB)
        self.loss_D_B = loss_D_B.data[0]
        
    def backward_D_C(self):
        fake_AC = self.fake_A_pool.query(self.fake_AC)
        loss_D_C = self.backward_D_basic(self.netD_C, self.real_A, fake_AC)
        self.loss_D_C = loss_D_C.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_AB = self.netG_AB(self.real_B)
            loss_idt_AB = self.criterionIdt(idt_AB, self.real_B) * lambda_B * lambda_idt
            idt_AC = self.netG_AC(self.real_C)
            loss_idt_AC = self.criterionIdt(idt_AC, self.real_C) * lambda_C * lambda_idt

            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            idt_C = self.netG_C(self.real_A)
            loss_idt_C = self.criterionIdt(idt_C, self.real_A) * lambda_A * lambda_idt

            self.idt_AB = idt_AB.data
            self.idt_AC = idt_AC.data
            self.idt_B = idt_B.data
            self.idt_C = idt_C.data
            self.loss_idt_AB = loss_idt_AB.data[0]
            self.loss_idt_AC = loss_idt_AC.data[0]
            self.loss_idt_B = loss_idt_B.data[0]
            self.loss_idt_C = loss_idt_C.data[0]
        else:
            loss_idt_AB = 0
            loss_idt_AC = 0
            loss_idt_B = 0
            loss_idt_C = 0
            self.loss_idt_AB = 0
            self,loss_idt_AC = 0
            self.loss_idt_B = 0
            self.loss_idt_C = 0

        # GAN loss D_A(G_A(A))
        fake_B = self.netG_AB(self.real_A)
        pred_fake = self.netD_AB(fake_B)
        loss_G_AB = self.criterionGAN(pred_fake, True)

        # GAN loss D_A(G_A(A))
        fake_C = self.netG_AC(self.real_A)
        pred_fake = self.netD_AC(fake_C)
        loss_G_AC = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_AB = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_AB)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # GAN loss D_C(G_C(C))
        fake_AC = self.netG_C(self.real_C)
        pred_fake = self.netD_C(fake_AC)
        loss_G_C = self.criterionGAN(pred_fake, True)


        # Forward cycle loss
        rec_AB = self.netG_B(fake_B)
        loss_cycle_AB = self.criterionCycle(rec_AB, self.real_A) * lambda_A

        rec_AC = self.netG_C(fake_C)
        loss_cycle_AC = self.criterionCycle(rec_AC, self.real_A) * lambda_A

        # Backward cycle loss
        rec_B = self.netG_AB(fake_AB)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        # Backward cycle loss
        rec_C = self.netG_AC(fake_AC)
        loss_cycle_C = self.criterionCycle(rec_C, self.real_C) * lambda_C

        # combined loss
        loss_G = loss_G_AB + loss_G_AC+ loss_G_B + loss_G_C + loss_cycle_AB + loss_cycle_AC + loss_cycle_B + loss_cycle_C + loss_idt_AB + loss_idt_AC + loss_idt_B + loss_idt_C
        loss_G.backward()

        self.fake_AB = fake_AB.data
        self.fake_AC = fake_AC.data
        self.fake_B = fake_B.data
        self.fake_C = fake_C.data
        
        self.rec_AB = rec_AB.data
        self.rec_AC = rec_AC.data
        self.rec_B = rec_B.data
        self.rec_C = rec_C.data

        self.loss_G_AB = loss_G_AB.data[0]
        self.loss_G_AC = loss_G_AC.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_G_C = loss_G_C.data[0]
        self.loss_cycle_AB = loss_cycle_AB.data[0]
        self.loss_cycle_AC = loss_cycle_AC.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        self.loss_cycle_C = loss_cycle_C.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_GB.zero_grad()
        self.optimizer_GC.zero_grad()
        self.backward_G()
        self.optimizer_GB.step()
        self.optimizer_GC.step()
        
        # D_A
        self.optimizer_D_AB.zero_grad()
        self.backward_D_AB()
        self.optimizer_D_AB.step()

        self.optimizer_D_AC.zero_grad()
        self.backward_D_AC()
        self.optimizer_D_AC.step()

        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step() 
        # D_C
        self.optimizer_D_C.zero_grad()
        self.backward_D_C()
        self.optimizer_D_C.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_AB', self.loss_D_AB), ('G_AB', self.loss_G_AB), ('Cyc_AB', self.loss_cycle_AB),
        	                     ('D_AC', self.loss_D_AC), ('G_AC', self.loss_G_AC), ('Cyc_AC', self.loss_cycle_AC),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B),
                                 ('D_C', self.loss_D_C), ('G_C', self.loss_G_C), ('Cyc_C',  self.loss_cycle_C)])
        if self.opt.identity > 0.0:
            ret_errors['idt_AB'] = self.loss_idt_AB
            ret_errors['idt_AC'] = self.loss_idt_AC
            ret_errors['idt_B'] = self.loss_idt_B
            ret_errors['idt_C'] = self.loss_idt_C
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        real_B = util.tensor2im(self.input_B)
        real_C = util.tensor2im(self.input_C)
        
        fake_AB = util.tensor2im(self.fake_AB)
        fake_AC = util.tensor2im(self.fake_AC)
        fake_B = util.tensor2im(self.fake_B)
        fake_C = util.tensor2im(self.fake_C)
        
        rec_AB = util.tensor2im(self.rec_AB)
        rec_AC = util.tensor2im(self.rec_AC)
        rec_B = util.tensor2im(self.rec_B)
        rec_C = util.tensor2im(self.rec_C)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_AB', fake_AB), ('rec_AB', rec_AB),
        	                       ('real_A', real_A), ('fake_AC', fake_AC), ('rec_AC', rec_AC),
                                   ('real_B', real_B), ('fake_B', fake_B), ('rec_B', rec_B),
                                   ('real_C', real_C), ('fake_C', fake_C), ('rec_C', rec_C)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_AB'] = util.tensor2im(self.idt_AB)
            ret_visuals['idt_AC'] = util.tensor2im(self.idt_AC)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
            ret_visuals['idt_C'] = util.tensor2im(self.idt_C)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.gpu_ids)
        self.save_network(self.netD_AB, 'D_AB', label, self.gpu_ids)
        self.save_network(self.netG_AC, 'G_AC', label, self.gpu_ids)
        self.save_network(self.netD_AC, 'D_AC', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netG_C, 'G_C', label, self.gpu_ids)
        self.save_network(self.netD_C, 'D_C', label, self.gpu_ids)

