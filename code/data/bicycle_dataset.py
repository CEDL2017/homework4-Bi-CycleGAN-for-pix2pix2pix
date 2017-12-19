import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class BicycleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_C = os.path.join(opt.dataroot, opt.phase + 'C') # johnson

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.C_paths = make_dataset(self.dir_C) # johnson

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.C_paths = sorted(self.C_paths) # johnson
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.C_size = len(self.C_paths) # johnson
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        index_A = index % self.A_size
        if self.opt.serial_batches:
            index_B = index % self.B_size
            index_C = index % self.C_size # johnson
        else:
            index_B = random.randint(0, self.B_size - 1)
            index_C = random.randint(0, self.C_size - 1) # johnson
        B_path = self.B_paths[index_B]
        C_path = self.C_paths[index_C] # johnson

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        C_img = Image.open(C_path).convert('RGB') # johnson

        A = self.transform(A_img)
        B = self.transform(B_img)
        C = self.transform(C_img) # johnson
        # johnson
        if self.opt.which_direction == 'AtoBtoC':
            input_nc = self.opt.input_nc
            intermediate_nc = self.opt.intermediate_nc
            output_nc = self.opt.output_nc
        elif self.opt.which_direction == 'BtoCtoA':
            input_nc = self.opt.intermediate_nc
            intermediate_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        elif self.opt.which_direction == 'CtoAtoB':
            input_nc = self.opt.output_nc
            intermediate_nc = self.opt.input_nc
            output_nc = self.opt.intermediate_nc
        else:
            raise ValueError("Invalid opt.which_direction {}".format(self.opt.which_direction))

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if intermediate_nc == 1: # RGB to gray, johnson
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray, johnson
            tmp = C[0, ...] * 0.299 + C[1, ...] * 0.587 + C[2, ...] * 0.114
            C = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'C': C,
                'A_paths': A_path, 'B_paths': B_path, 'C_paths': C_path} # johnson

    def __len__(self):
        return max(self.A_size, self.B_size, self.C_size) # johnson

    def name(self):
        return 'UnalignedDataset'
