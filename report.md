# 江愷笙 <span style="color:black">(106062568)</span>

Here is the [github page]() of my report.

# Homework4 report

## Overview

This project is related to
* Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros, "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017
* Jun-Yan Zhu, Richard Zhang, Deepak Pathak, Trevor Darrell, Alexei A. Efros, Oliver Wang, Eli Shechtman, "Toward Multimodal Image-to-Image Translation"
* Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo, "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation"

>With CycleGAN, we can better transfer the styles of images from domain A to domain B and vice versa without unpaired data. What will it be if we have two cycles (A, B) and (B, C) where B is the common modality shared by the two cycles?

## Implementation

In this project we have to find a dataset and use the idea of cycle-GAN to do the style transfer. The main idea is to train two cycle-GAN, which is A -> B -> A' and B -> C -> B'. Here I found some interest from "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation", which is a pretty new paper with amazing results. I use one of the dataset which they have used in their work called "CelebA", which collects many different images with people in different style, and label them with 40 attributes such as Black_Hair, Male, Brown_Hair, Pale, Chubby etc. Since each image may have multiple attributes, for example, a people with black is probably a man, so I choose three domain that are Black_Hair, Blond_Hair and Gray_Hair in order to let each image only occur in one domain.

### get_CelebA.py

In order to use the dataset and split it into the three domain I want, I write my own code to read the label and save the image to the related folder.

First we define the image_path and metadata_path.

```python
image_path = '/home/petersci/CycleGAN_TensorFlow/data/CelebA_nocrop/images'
metadata_path = '/home/petersci/CycleGAN_TensorFlow/data/list_attr_celeba.txt'
```
And then we read the attributes in the file and do the shuffling.

```python
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

lines = lines[2:]
random.shuffle(lines)   # random shuffling
```
Finally, we check if this is the attribute that we want, split it into training and testing data, and save the image to the saving_path.

```python
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

test_filenames_black = filenames_black[0:int(len(filenames_black)/10)]
train_filenames_black = filenames_black[int(len(filenames_black)/10):]

print('number of training Black_Hair is ', len(train_filenames_black))
print('number of testing Black_Hair is ', len(test_filenames_black))

for idx, path in enumerate(train_filenames_black):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Black_Hair/'+str(idx)+'.jpg')

for idx, path in enumerate(test_filenames_black):
    image = Image.open(path)
    image.save('/home/petersci/CycleGAN_TensorFlow/input/CelebA/Black_Hair_test/'+str(idx)+'.jpg')
```

## Installation

### Dependencies

* Tensorflow
* Python3.5

### Getting Started

* For the starter code, I use Tensorflow implementation of CycleGANs by Harry Yang.
* To get started, first clone the repository and renamed it as CycleGAN_TensorFlow.
* Use `$ bash ./download_datasets.sh horse2zebra` to download horse2zebra dataset for testing.
* Use `$ bash download.sh` to download CelebA.
* To split the data into the domain we want, Use `$ python get_CelebA.py`, and don't forget to change image_path, metadata_path and saving_path in get_CelebA.py to your own directory.
* Use `$ convert '*.jpg[128x128!]' resize%06d.jpg` to resize the image in the folder to the size you want.
* Use `$ for i in $(seq num1 num2); do rm $i.jpg; done` to delete the unresized image.
* Follow the instruction [here](https://github.com/leehomyc/cyclegan-1) for training and testing.
* If you want to test my result, use the checkpoint in "black2blond_ckpt" and "blond2gray_ckpt" and do the testing.

## Results

### What scenario do I apply in?
you are encouraged to elaborate the motivation here

### What do I modify? 
you can show some snippet

### Qualitative results
put some interesting images generated from your Bi-CycleGANs

### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

### Others

### Reference
