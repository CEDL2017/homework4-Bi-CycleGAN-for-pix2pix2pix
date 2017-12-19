# Homework4 Report 
#### 106062581 吳奕萱

## What scenario do I apply in?

#### Basic setting
- A : Monet painting
- B : real life photo
- C : Ukiyoe painting (浮世繪)
- Two types of CycleGAN
    - A <-> B
    - B <-> C

#### Why I chose these domains
In real life, drawing is a tough skill that have to practice a lot and much depends on gift.
Thus, by implement auto-style-transfer algorithm, we can achieve following benefits
1. Art to photo: Learn composition of visual art from outstranding artist
2. Photo to art: Create astonishing art work without any domain knowledge and without put ourself through the art mill
3. Art to another art-style: Learn the invariance between two arts which make art BIG

## What do I modify?
I didn't modify a lot in the basic CycleGAN structure and didn't success to get benefit infomation between two cycles.
Main model is based on tensorflow implemetation from [XhuJoy](https://github.com/xhujoy/CycleGAN-tensorflow).
And choose the dataset monet2photo and ukiyoe2photo.

## Qualitative results
### monet to photo
<table>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/01120.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/AtoB_01120.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/00360.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/AtoB_00360.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/00700.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/AtoB_00700.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/00020.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/AtoB_00020.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/00030.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/AtoB_00030.jpg'></td></tr>
</table>
### photo to monet
<table>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB_origin/2014-08-07 21:31:39.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/BtoA_2014-08-07 21:31:39.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB_origin/2014-10-25 23:33:01.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/BtoA_2014-10-25 23:33:01.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB_origin/2014-12-21 04:02:05.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/BtoA_2014-12-21 04:02:05.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB_origin/2014-09-25 08:57:42.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/BtoA_2014-09-25 08:57:42.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB_origin/2015-04-27 20:00:53.jpg'></td><td><img src='./CycleGAN-tensorflow/test/monetResult/BtoA_2015-04-27 20:00:53.jpg'></td></tr>
</table>

### ukiyoe to photo
<table>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01340.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/AtoB_01340.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01349.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/AtoB_01349.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01341.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/AtoB_01341.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01224.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/AtoB_01224.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01202.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/AtoB_01202.jpg'></td></tr>
</table>

### photo to ukiyoe
<table>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB_origin/2015-04-09 09:55:16.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/BtoA_2015-04-09 09:55:16.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB_origin/2014-08-03 09:47:19.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/BtoA_2014-08-03 09:47:19.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB_origin/2014-08-08 17:06:41.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/BtoA_2014-08-08 17:06:41.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB_origin/2015-04-12 08:45:07.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/BtoA_2015-04-12 08:45:07.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB_origin/2014-10-20 03:38:37.jpg'></td><td><img src='./CycleGAN-tensorflow/test/ukiyoeResult/BtoA_2014-10-20 03:38:37.jpg'></td></tr>
</table>

### ukiyoe to photo to monet
<table>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01202.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB/AtoB_01202.jpg'></td><td><img src='./CycleGAN-tensorflow/test/m2k/BtoA_AtoB_01202.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01224.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB/AtoB_01224.jpg'></td><td><img src='./CycleGAN-tensorflow/test/m2k/BtoA_AtoB_01224.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01340.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB/AtoB_01340.jpg'></td><td><img src='./CycleGAN-tensorflow/test/m2k/BtoA_AtoB_01340.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01341.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB/AtoB_01341.jpg'></td><td><img src='./CycleGAN-tensorflow/test/m2k/BtoA_AtoB_01341.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testA_origin/01349.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testB/AtoB_01349.jpg'></td><td><img src='./CycleGAN-tensorflow/test/m2k/BtoA_AtoB_01349.jpg'></td></tr>
</table>

### monet to photo to ukiyoe
<table>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/00020.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB/AtoB_00020.jpg'></td><td><img src='./CycleGAN-tensorflow/test/k2m/BtoA_AtoB_00020.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/00030.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB/AtoB_00030.jpg'></td><td><img src='./CycleGAN-tensorflow/test/k2m/BtoA_AtoB_00030.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/00360.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB/AtoB_00360.jpg'></td><td><img src='./CycleGAN-tensorflow/test/k2m/BtoA_AtoB_00360.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/00700.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB/AtoB_00700.jpg'></td><td><img src='./CycleGAN-tensorflow/test/k2m/BtoA_AtoB_00700.jpg'></td></tr>
<tr><td><img src='./CycleGAN-tensorflow/datasets/monet2photo/testA_origin/01120.jpg'></td><td><img src='./CycleGAN-tensorflow/datasets/ukiyoe2photo/testB/AtoB_01120.jpg'></td><td><img src='./CycleGAN-tensorflow/test/k2m/BtoA_AtoB_01120.jpg'></td></tr>
</table>

## My thoughts
### Pros
1. the monet part to really good work on both direction
2. I think some of results are comparible to real works

### Cons
1. The style of ukiyoe art is so strong that GAN structure cannot mapping things between photo and ukiyoe which make the result not so convincive.
    - Solution: the photos in the dataset are basicly nature views. To solve the mapping problem, we can try to put more human picture (there are more human paintings in ukiyoe) in the data which make GAN learn better mapping.
2. The timing time of two CycleGAN is twice of single CycleGAN. But there should be some shared invariance of these two CycleGAN.
    - Solution: the naive solution of this problem is to initial weights of second CycleGAN by the first-trained one. This may help second-trained CycleGAN to converge faster.
    
## Others
### Reference
- [Tensorflow implementation](https://github.com/xhujoy/CycleGAN-tensorflow)
- [CycleGAN paper](https://arxiv.org/abs/1703.10593)