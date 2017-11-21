# Homework4-Bi-CycleGAN

In this homework, you will need to extend the idea of [CycleGAN](https://arxiv.org/abs/1703.10593) to *multiple-domain* scenarios. We may call these types of models: Bi-CycleGANs or Tri-CycleGANs, or even Multi-CycleGANs.


## Introduction

As mentioned in the original paper, CycleGAN is designed for **Unpaired Image-to-Image Translation**. Many interesting applications of CycleGAN can be found in the paper and on their project [website](https://junyanz.github.io/CycleGAN/). 

Motivation: with CycleGAN, we can better transfer the styles of images from domain A to domain B and vice versa. What will it be if we have two cycles (A, B) and (A, C) where A is the common domain shared by the two cycles? Your task is to find some scenarios and related datasets that can demonstrate the idea and the advantages of Bi-CycleGAN. :fire:

For confused students, here are some concrete examples:

>Cycle 1: A ←→ B   
>Cycle 2: A ←→ C

A is the common domain (or common modality) shared by the two cycles.

Examples:   
>[Automatically generated dataset]   
>A: MNIST   
>B: Inverted MNIST (black->white, white->black)   
>C: Red-Green MNIST (black->red, white->green)   

Or

>A: An RGB image containing a person   
>B: Depth map   
>C: Keypoints (body joints)   

Or

>A: object captured from 0 degree    
>B: object captured from 30 degree   
>C: object captured from 60 degree   
>[Can be obtained by ShapeNet]

More other examples might be inspired by [here](https://github.com/mingyuliutw/UNIT)

**Note that the relation (correspondence) among multiple modalities is not provided (if you generate the dataset yourselves, you need to shuffle the dataset. Although it's still implicitly paired, it's fine for this homework)**

## Official implementation of CycleGAN
- Tensorflow implementation: [here](https://github.com/junyanz/CycleGAN)
- Pytorch implementation: [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)


## TODO
- [75%] Find a scenario and related datasets to demonstrate the idea of BicycleGAN. 
  - You may simply train two CycleGANs separately using the original code of CycleGAN without enforcing the consistency of the shared domain between the two cycles. 
- [15%] Implement Bi-CycleGANs so that you can jointly train the two cycles with additional constraints on the consistency of the shared domain. Compare the results of separate training and joint training.
- [10%] Report 
- [5%] Bonus, share you code and what you learn on github or  yourpersonal blogs, such as [this](https://andrewliao11.github.io/object/detection/2016/07/23/detection/)

## Other
- Deadline: Dec. 7 23:59, 2017
- Office hour by appointment [Yuan-Hong Liao](https://andrewliao11.github.io).
- Contact *andrewliao11@gmail.com* for bugs report or any questions.
