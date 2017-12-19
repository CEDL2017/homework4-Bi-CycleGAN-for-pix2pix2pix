import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import copy
import pdb
opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
#copy
opt2 = copy.copy(opt)
opt2.dataroot = opt.dataroot_more

data_loader = CreateDataLoader(opt)
data_loader2 = CreateDataLoader(opt2)
dataset_size = min(len(data_loader),len(data_loader2))

dataset = data_loader.load_data()
dataset2 = data_loader2.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test
loader = enumerate(dataset)
loader2 = enumerate(dataset2)
for steps in range(dataset_size):
    i, data = next(loader)
    i, data2 = next(loader2)
    
#for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data, data2)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()
