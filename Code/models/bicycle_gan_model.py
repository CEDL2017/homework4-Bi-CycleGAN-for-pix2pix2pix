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


class BiCycleGANModel(BaseModel):
    def name(self):
        return 'BiCycleGANModel'

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
        self.netG_BA = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_BC = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_CB = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_C = networks.define_D(opt.output_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_AB, 'G_AB', which_epoch)
            self.load_network(self.netG_BA, 'G_BA', which_epoch)
            self.load_network(self.netG_BC, 'G_BC', which_epoch)
            self.load_network(self.netG_CB, 'G_CB', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
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
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_AB.parameters(), self.netG_BA.parameters(), self.netG_BC.parameters(), self.netG_CB.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_C)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_AB)
        networks.print_network(self.netG_BA)
        networks.print_network(self.netG_BC)
        networks.print_network(self.netG_CB)
        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_C)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_C = input['C' if AtoB else 'B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        fake_AB = self.netG_AB(real_A)
        self.rec_ABA = self.netG_BA(fake_AB).data
        self.rec_ABC = self.netG_BC(fake_AB).data
        self.fake_AB = fake_AB.data

        real_B = Variable(self.input_B, volatile=True)
        fake_BA = self.netG_BA(real_B)
        self.rec_BAB = self.netG_AB(fake_BA).data
        self.fake_BA = fake_BA.data
        
        real_B = Variable(self.input_B, volatile=True)
        fake_BC = self.netG_BC(real_B)
        self.rec_BCB = self.netG_CB(fake_BC).data
        self.fake_BC = fake_BC.data
        
        real_C = Variable(self.input_C, volatile=True)
        fake_CB = self.netG_CB(real_C)
        self.rec_CBC = self.netG_BC(fake_CB).data
        self.rec_CBA = self.netG_BA(fake_CB).data
        self.fake_CB = fake_CB.data

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

    def backward_D_A(self):
        fake_BA = self.fake_A_pool.query(self.fake_BA)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_BA)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_AB = self.fake_B_pool.query(self.fake_AB)
        loss_D_AB = self.backward_D_basic(self.netD_B, self.real_B, fake_AB)
        
        fake_CB = self.fake_B_pool.query(self.fake_CB)
        loss_D_CB = self.backward_D_basic(self.netD_B, self.real_B, fake_CB)
        self.loss_D_B = (loss_D_AB.data[0] + loss_D_CB.data[0]) * 0.5
        
    def backward_D_C(self):
        fake_BC = self.fake_B_pool.query(self.fake_BC)
        loss_D_C = self.backward_D_basic(self.netD_C, self.real_C, fake_BC)
        self.loss_D_C = loss_D_C.data[0]

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_AB should be identity if real_B is fed.
            idt_AB = self.netG_AB(self.real_B)
            loss_idt_AB = self.criterionIdt(idt_AB, self.real_B) * lambda_B * lambda_idt
            # G_BA should be identity if real_A is fed.
            idt_BA = self.netG_BA(self.real_A)
            loss_idt_BA = self.criterionIdt(idt_BA, self.real_A) * lambda_A * lambda_idt
            # G_BC should be identity if real_A is fed.
            idt_BC = self.netG_BC(self.real_C)
            loss_idt_BC = self.criterionIdt(idt_BC, self.real_C) * lambda_A * lambda_idt
            # G_CB should be identity if real_A is fed.
            idt_CB = self.netG_CB(self.real_B)
            loss_idt_CB = self.criterionIdt(idt_CB, self.real_B) * lambda_B * lambda_idt

            self.idt_AB = idt_AB.data
            self.idt_BA = idt_BA.data
            self.idt_BC = idt_BC.data
            self.idt_CB = idt_CB.data
            self.loss_idt_AB = loss_idt_AB.data[0]
            self.loss_idt_BA = loss_idt_BA.data[0]
            self.loss_idt_BC = loss_idt_BC.data[0]
            self.loss_idt_CB = loss_idt_CB.data[0]
        else:
            loss_idt_AB = 0
            loss_idt_BA = 0
            loss_idt_BC = 0
            loss_idt_CB = 0
            self.loss_idt_AB = 0
            self.loss_idt_BA = 0
            self.loss_idt_BC = 0
            self.loss_idt_CB = 0

        # GAN loss D_B(G_AB(A))
        fake_AB = self.netG_AB(self.real_A)
        pred_fake = self.netD_B(fake_AB)
        loss_G_AB = self.criterionGAN(pred_fake, True)

        # GAN loss D_A(G_BA(B))
        fake_BA = self.netG_BA(self.real_B)
        pred_fake = self.netD_A(fake_BA)
        loss_G_BA = self.criterionGAN(pred_fake, True)
        # GAN loss D_C(G_BC(B))
        fake_BC = self.netG_BC(self.real_B)
        pred_fake = self.netD_C(fake_BC)
        loss_G_BC = self.criterionGAN(pred_fake, True)
        
        # GAN loss D_B(G_CB(C))
        fake_CB = self.netG_CB(self.real_C)
        pred_fake = self.netD_B(fake_CB)
        loss_G_CB = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_ABA = self.netG_BA(fake_AB)
        loss_cycle_ABA = self.criterionCycle(rec_ABA, self.real_A) * lambda_A
        rec_CBA = self.netG_BA(self.netG_CB(self.netG_BC(fake_AB)))
        loss_cycle_CBA = self.criterionCycle(rec_CBA, self.real_A) * lambda_A

        # Backward cycle loss
        rec_BAB = self.netG_AB(fake_BA)
        loss_cycle_BAB = self.criterionCycle(rec_BAB, self.real_B) * lambda_B
        rec_BCB = self.netG_CB(fake_BC)
        loss_cycle_BCB = self.criterionCycle(rec_BCB, self.real_B) * lambda_B
        
        # Forward cycle loss
        rec_ABC = self.netG_BC(self.netG_AB(self.netG_BA(fake_CB)))
        loss_cycle_ABC = self.criterionCycle(rec_ABC, self.real_C) * lambda_A
        rec_CBC = self.netG_BC(fake_CB)
        loss_cycle_CBC = self.criterionCycle(rec_CBC, self.real_C) * lambda_A
        
        # combined loss
        loss_G = loss_G_AB + loss_G_BA + loss_G_BC + loss_G_CB + \
            loss_cycle_ABA + loss_cycle_CBA + loss_cycle_BCB + loss_cycle_BAB + loss_cycle_ABC + loss_cycle_CBC + \
            loss_idt_AB + loss_idt_BA + loss_idt_BC + loss_idt_CB
        loss_G.backward()

        self.fake_AB = fake_AB.data
        self.fake_CB = fake_CB.data
        self.fake_BA = fake_BA.data
        self.fake_BC = fake_BC.data
        self.rec_ABA = rec_ABA.data
        self.rec_CBA = rec_CBA.data
        self.rec_BAB = rec_BAB.data
        self.rec_BCB = rec_BCB.data
        self.rec_ABC = rec_ABC.data
        self.rec_CBC = rec_CBC.data

        self.loss_G_AB = loss_G_AB.data[0]
        self.loss_G_BA = loss_G_BA.data[0]
        self.loss_G_BC = loss_G_BC.data[0]
        self.loss_G_CB = loss_G_CB.data[0]
        self.loss_cycle_ABA = loss_cycle_ABA.data[0]
        self.loss_cycle_CBA = loss_cycle_CBA.data[0]
        self.loss_cycle_BAB = loss_cycle_BAB.data[0]
        self.loss_cycle_BCB = loss_cycle_BCB.data[0]
        self.loss_cycle_ABC = loss_cycle_ABC.data[0]
        self.loss_cycle_CBC = loss_cycle_CBC.data[0]

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_AB and G_BA and G_BC and G_CB
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
        # D_C
        self.optimizer_D_C.zero_grad()
        self.backward_D_C()
        self.optimizer_D_C.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_AB', self.loss_G_AB), 
                                 ('Cyc_ABA', self.loss_cycle_ABA), ('Cyc_CBA', self.loss_cycle_CBA),
                                 ('D_B', self.loss_D_B), ('G_BA', self.loss_G_BA), ('G_BC', self.loss_G_BC), 
                                 ('Cyc_BAB',  self.loss_cycle_BAB), ('Cyc_BCB',  self.loss_cycle_BCB),
                                 ('D_C', self.loss_D_C), ('G_CB', self.loss_G_CB), 
                                 ('Cyc_ABC',  self.loss_cycle_ABC), ('Cyc_CBC',  self.loss_cycle_CBC)])
        if self.opt.identity > 0.0:
            ret_errors['idt_AB'] = self.loss_idt_AB
            ret_errors['idt_BA'] = self.loss_idt_BA
            ret_errors['idt_BC'] = self.loss_idt_BC
            ret_errors['idt_CB'] = self.loss_idt_CB
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_AB = util.tensor2im(self.fake_AB)
        rec_ABA = util.tensor2im(self.rec_ABA)
        rec_ABC = util.tensor2im(self.rec_ABC)
        
        real_B = util.tensor2im(self.input_B)
        fake_BA = util.tensor2im(self.fake_BA)
        rec_BAB = util.tensor2im(self.rec_BAB)
        fake_BC = util.tensor2im(self.fake_BC)
        rec_BCB = util.tensor2im(self.rec_BCB)
        
        real_C = util.tensor2im(self.input_C)
        fake_CB = util.tensor2im(self.fake_CB)
        rec_CBA = util.tensor2im(self.rec_CBA)
        rec_CBC = util.tensor2im(self.rec_CBC)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_AB', fake_AB), ('rec_ABA', rec_ABA), ('rec_ABC', rec_ABC),
                                   ('real_B', real_B), ('fake_BA', fake_BA), ('rec_BAB', rec_BAB),
                                                       ('fake_BC', fake_BC), ('rec_BCB', rec_BCB),
                                   ('real_C', real_C), ('fake_CB', fake_CB), ('rec_CBC', rec_CBC), ('rec_CBA', rec_CBA)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_AB'] = util.tensor2im(self.idt_AB)
            ret_visuals['idt_BA'] = util.tensor2im(self.idt_BA)
            ret_visuals['idt_BC'] = util.tensor2im(self.idt_BC)
            ret_visuals['idt_CB'] = util.tensor2im(self.idt_CB)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_AB, 'G_AB', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_BA, 'G_BA', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netG_BC, 'G_BC', label, self.gpu_ids)
        self.save_network(self.netG_CB, 'G_CB', label, self.gpu_ids)
        self.save_network(self.netD_C, 'D_C', label, self.gpu_ids)
