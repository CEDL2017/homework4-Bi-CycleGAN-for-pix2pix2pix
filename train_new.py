import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import cv2
import copy
import pdb

opt = TrainOptions().parse()
opt2 = copy.copy(opt)
# change some opts
opt2.dataroot = opt.dataroot_more
#opt.which_direction = 'BtoA' # ensure the common domain A is photo
#opt2.which_direction = 'BtoA'
data_loader = CreateDataLoader(opt)
data_loader2 = CreateDataLoader(opt2)

dataset = data_loader.load_data()
dataset2 = data_loader2.load_data()
dataset_size = min(len(data_loader),len(data_loader2))
#dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
model = create_model(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    #for i, data in enumerate(dataset):
    loader = enumerate(dataset)
    loader2 = enumerate(dataset2)
    for step_in_epoch in range(dataset_size):
        i, data = next(loader)
        i, data2 = next(loader2)
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data, data2)
        
        #cv2.imwrite('A.jpg',(model.input_A[0].cpu().numpy().transpose(1,2,0)+1)/2.*255) # check
        #pdb.set_trace()
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()
