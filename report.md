# **Homework4 Bi-CycleGAN for Image-to-Image-to-Image Translation**

## **What scenario do I apply in?**
* Use Cycle-GAN to implement Image-to-Image translation
* My subject : horse -> zebra -> donkey

### **Dataset**
* Use [icrawler](https://github.com/hellock/icrawler) to collect enough donkey dataset from Google and Flickr, and resize all images to 256*256 ( Download lots of high-quality images from Flickr!!) 
* Select 500 images for testing

### Process
1. First, I train CycleGAN on horse to zebra, and then turn zebra to donkey.
2. So, I concatenate these two phases into one, then we can called it Bi-CycleGAN 

### Qualitative results

# **Cycle-GAN**
* horse <-> zebra

| real-horse | fake-zebra | rec-horse | real-zebra | fake-horse | rec-zebra | 
| -------- | -------- | -------- | -------- | -------- | -------- |
|  ![](https://i.imgur.com/y3oe6Sz.png) | ![](https://i.imgur.com/iTQZlV0.png)| ![](https://i.imgur.com/vZDmRLQ.png)| ![](https://i.imgur.com/DILa4SJ.png)| ![](https://i.imgur.com/n8TM2qi.png)| ![](https://i.imgur.com/HkGyQWw.png)|
| ![](https://i.imgur.com/BRNLzEa.png)| ![](https://i.imgur.com/qZqqo8c.png)|![](https://i.imgur.com/woigZoc.png)| ![](https://i.imgur.com/dAoqvcm.png)|![](https://i.imgur.com/mZmaRLb.png)| ![](https://i.imgur.com/wwKr02e.png)|


* zebra <-> donkey

| real-zebra | fake-donkey | rec-zebra | real-donkey | fake-zebra | rec-donkey | 
| -------- | -------- | -------- | -------- | -------- | -------- |
|![](https://i.imgur.com/FrqEOlb.png)|![](https://i.imgur.com/bCqQGpZ.png)|![](https://i.imgur.com/oCLHorL.png)|![](https://i.imgur.com/E7YQTeO.png)|![](https://i.imgur.com/yD3dxx9.png)|![](https://i.imgur.com/Yrqhsez.png)|
|![](https://i.imgur.com/z6QkhUQ.png)|![](https://i.imgur.com/yfPuWcz.png)|![](https://i.imgur.com/YFfoWFD.png)|![](https://i.imgur.com/EuXI7tM.png)|![](https://i.imgur.com/Rv2NrtO.png)|![](https://i.imgur.com/RLvEIPK.png)

1. It seems like the fake donkey has both the flurry feature from donkey and the line feature form zebra, so the output is not good enough, comparing to the output of horse to zebra. 
2. The fake-zebra are more likely to be poetic,comparing to the real-horse.

# **Bi-CycleGAN**
#### **Modify**
We have two generaters and two discriminaters in each cycle. I use the discriminators of two cycles in Domain A to train together and finetune parameters from pretrained CycleGAN.

* horse -> zebra -> donkey

| horse | zebra | donkey |
| -------- | -------- | -------- |
| ![](https://i.imgur.com/fQeYT5V.png)| ![](https://i.imgur.com/9p54PvH.png)| ![](https://i.imgur.com/UaX8aLI.png)|

The results of Bi-CycleGAN are more blurred than CycleGAN.



## My thoughts

1. When the image has complicated background, the fake image will be a worst output.
2. Just like the fake-horse, the background is also turn to gray, so there are still many defects have to be solved.
3. But for most of the clear image, CycleGAN still can learn the most important feature from Domain A.

## Reference

1. [CycleGAN](https://github.com/Cupido10/pytorch-CycleGAN-and-pix2pix)
2. [Bi-CycleGAN](https://github.com/junyanz/BicycleGAN)
3. [icrawler](https://github.com/hellock/icrawler)
 