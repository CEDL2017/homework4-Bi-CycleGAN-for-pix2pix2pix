# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 03:55:09 2017

@author: Peter
"""

import os
import random
from PIL import Image


image_path = '/home/petersci/CycleGAN_TensorFlow/data/CelebA_nocrop/images'
metadata_path = '/home/petersci/CycleGAN_TensorFlow/data/list_attr_celeba.txt'

lines = open(metadata_path, 'r').readlines()
num_data = int(lines[0])
attr2idx = {}
idx2attr = {}

print ('Start preprocessing dataset..!')

attrs = lines[1].split()
for i, attr in enumerate(attrs):
    attr2idx[attr] = i
    idx2attr[i] = attr

#selected_attrs = ['Black_Hair', 'Blond_Hair', 'Gray_Hair', 'Attractive', 'Bags_Under_Eyes', 'Chubby']
filenames_black = []
filenames_blond = []
filenames_gray = []
filenames_attract = []
filenames_eyebags = []
filenames_chubby = []

lines = lines[2:]
random.shuffle(lines)   # random shuffling
for i, line in enumerate(lines):

    splits = line.split()
    filename = splits[0]
    values = splits[1:]

    label = []
    for idx, value in enumerate(values):
        attr = idx2attr[idx]

        if attr == 'Black_Hair':
            if value == '1':
                filenames_black.append(os.path.join(image_path, filename))
        if attr == 'Blond_Hair':
            if value == '1':
                filenames_blond.append(os.path.join(image_path, filename))
        if attr == 'Gray_Hair':
            if value == '1':
                filenames_gray.append(os.path.join(image_path, filename))
        if attr == 'Attractive':
            if value == '1':
                filenames_attract.append(os.path.join(image_path, filename))
        if attr == 'Bags_Under_Eyes':
            if value == '1':
                filenames_eyebags.append(os.path.join(image_path, filename))
        if attr == 'Chubby':
            if value == '1':
                filenames_chubby.append(os.path.join(image_path, filename))
        
test_filenames_black = filenames_black[0:int(len(filenames_black)/10)]
test_filenames_blond = filenames_blond[0:int(len(filenames_blond)/10)]
test_filenames_gray = filenames_gray[0:int(len(filenames_gray)/10)]
test_filenames_attract = filenames_attract[0:int(len(filenames_attract)/10)]
test_filenames_eyebags = filenames_eyebags[0:int(len(filenames_eyebags)/10)]
test_filenames_chubby = filenames_chubby[0:int(len(filenames_chubby)/10)]

train_filenames_black = filenames_black[int(len(filenames_black)/10):]
train_filenames_blond = filenames_blond[int(len(filenames_blond)/10):]
train_filenames_gray = filenames_gray[int(len(filenames_gray)/10):]
train_filenames_attract = filenames_attract[int(len(filenames_attract)/10):]
train_filenames_eyebags = filenames_eyebags[int(len(filenames_eyebags)/10):]
train_filenames_chubby = filenames_chubby[int(len(filenames_chubby)/10):]

print('number of training Black_Hair is ', len(train_filenames_black))
print('number of testing Black_Hair is ', len(test_filenames_black))

print('number of training Blond_Hair is ', len(train_filenames_blond))
print('number of testing Blond_Hair is ', len(test_filenames_blond))

print('number of training Gray_Hair is ', len(train_filenames_gray))
print('number of testing Gray_Hair is ', len(test_filenames_gray))

print('number of training Attractive is ', len(train_filenames_attract))
print('number of testing Attractive is ', len(test_filenames_attract))

print('number of training Bags_Under_Eyes is ', len(train_filenames_eyebags))
print('number of testing Bags_Under_Eyes is ', len(test_filenames_eyebags))

print('number of training Chubby is ', len(train_filenames_chubby))
print('number of testing Chubby is ', len(test_filenames_chubby))

for idx, path in enumerate(train_filenames_black):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Black_Hair/'+str(idx)+'.jpg')
for idx, path in enumerate(train_filenames_blond):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Blond_Hair/'+str(idx)+'.jpg')
for idx, path in enumerate(train_filenames_gray):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Gray_Hair/'+str(idx)+'.jpg')
for idx, path in enumerate(train_filenames_attract):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Attractive/'+str(idx)+'.jpg')
for idx, path in enumerate(train_filenames_eyebags):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Bags_Under_Eyes/'+str(idx)+'.jpg')
for idx, path in enumerate(train_filenames_chubby):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Chubby/'+str(idx)+'.jpg')

for idx, path in enumerate(test_filenames_black):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Black_Hair_test/'+str(idx)+'.jpg')
for idx, path in enumerate(test_filenames_blond):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Blond_Hair_test/'+str(idx)+'.jpg')
for idx, path in enumerate(test_filenames_gray):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Gray_Hair_test/'+str(idx)+'.jpg')
for idx, path in enumerate(test_filenames_attract):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Attractive_test/'+str(idx)+'.jpg')
for idx, path in enumerate(test_filenames_eyebags):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Bags_Under_Eyes_test/'+str(idx)+'.jpg')
for idx, path in enumerate(test_filenames_chubby):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Chubby_test/'+str(idx)+'.jpg')

print ('Finished preprocessing dataset..!')


