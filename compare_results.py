import numpy as np
import os
from skimage.io import imread
from skimage.measure import compare_nrmse, compare_psnr, compare_ssim

def compare_results(im_true, im_gen): # between gen and real
	nrmse = compare_nrmse(im_true, im_gen)
	ssim = compare_ssim(im_true, im_gen, multichannel=True)
	psnr = compare_psnr(im_true, im_gen)
	return nrmse, ssim, psnr

def read_data(style_name, gen_direction):
	if gen_direction == 0: # photo2style
		true_imgs_path = os.path.join('data', style_name+'2photo', 'testB')
		direction_name = 'photo2' + style_name
	else: # style2photo
		true_imgs_path = os.path.join('data', style_name+'2photo', 'testA')
		direction_name = style_name + '2photo'
	gen_imgs_path = os.path.join('results', direction_name)
	
	true_imgs_name = os.listdir(true_imgs_path)
	num_imgs = len(true_imgs_name)
	nrmse = np.zeros((num_imgs,))
	ssim = np.zeros((num_imgs,))
	psnr = np.zeros((num_imgs,))
	wf_results = open('compare_results.txt', 'w')
	for i in range(num_imgs):
		im_true_path = os.path.join(true_imgs_path, true_imgs_name[i])
		im_true = imread(im_true_path)
		im_gen_path = os.path.join(gen_imgs_path, direction_name+'_'+true_imgs_name[i])
		im_gen = imread(im_gen_path)
		nrmse[i], ssim[i], psnr[i] = compare_results(im_true, im_gen)
		#print('[%s] %s: %10.5f %10.5f %10.5f' %(direction_name, true_imgs_name[i], nrmse[i], ssim[i], psnr[i]))
		wf_results.write('[%s] %s: %10.5f %10.5f %10.5f\n' %(direction_name, true_imgs_name[i], nrmse[i], ssim[i], psnr[i]))
	print('[%s] Avg.: %10.5f %10.5f %10.5f' %(direction_name, nrmse.mean(), ssim.mean(), psnr.mean()))
	wf_results.write('[%s] Avg.: %10.5f %10.5f %10.5f\n' %(direction_name, nrmse.mean(), ssim.mean(), psnr.mean()))
	wf_results.close()



read_data('monet', 0)
read_data('monet', 1)
read_data('ukiyoe', 0)
read_data('ukiyoe', 1)
