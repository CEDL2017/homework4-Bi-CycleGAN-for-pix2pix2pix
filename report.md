# Homework4 report

## Bi-Cycle GAN (and cyle GAN[1]): multiple-domain scenarios


### 105061583 賴承薰



## Scenario

<img src="We_bare_bears.png" width="70%"/>

#### ↑ Inspired by this cartoon show

Domain A: giant panda bear
> About 800 Pictures, manually captured from flickr

Domain B: polar bear
> Data are obtained from the database AWA2[2]

Domain C: grizzly bear
> Same as domain B


## Modifying 
Simply copy and append the part with domain B and modifd into C.

Most of the differene are modified in the file `./models/cycle_gan.py` and `./data/unaligned_dataset.py`

Due to the device limitation, my Bi-cycle GAN fail to complete 200 epochs. 

### Environment construction
* python 2.7
* pytorch 0.2.0_3 (macOS or Linux)
* CUDA

installation tutorial: `http://pytorch.org/`

## Qualitative results

### Ordinary cycle GAN

#### Epoch 50

<table border=1>
<tr>
<td>
<img src="./checkpoints/cyclegan_bear_2/images/epoch050_fake_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_2/images/epoch050_real_B.png" width="40%"/>
</td>
<td>
<img src="./checkpoints/cyclegan_bear_2/images/epoch050_real_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_2/images/epoch050_fake_B.png" width="40%"/>  
</td>
</tr>
<tr>
<td>
<img src="./checkpoints/cyclegan_bear_3/images/epoch050_fake_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_3/images/epoch050_real_B.png" width="40%"/>
</td>
<td>
<img src="./checkpoints/cyclegan_bear_3/images/epoch050_real_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_3/images/epoch050_fake_B.png" width="40%"/>  
</td>
</tr>


#### Epoch 100

<table border=1>
<tr>
<td>
<img src="./checkpoints/cyclegan_bear_2/images/epoch100_fake_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_2/images/epoch100_real_B.png" width="40%"/>
</td>
<td>
<img src="./checkpoints/cyclegan_bear_2/images/epoch100_real_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_2/images/epoch100_fake_B.png" width="40%"/>  
</td>
</tr>
<tr>
<td>
<img src="./checkpoints/cyclegan_bear_3/images/epoch100_fake_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_3/images/epoch100_real_B.png" width="40%"/>
</td>
<td>
<img src="./checkpoints/cyclegan_bear_3/images/epoch100_real_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_3/images/epoch100_fake_B.png" width="40%"/>  
</td>
</tr>

#### Epoch 150

<table border=1>
<tr>
<td>
<img src="./checkpoints/cyclegan_bear_2/images/epoch150_fake_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_2/images/epoch150_real_B.png" width="40%"/>
</td>
<td>
<img src="./checkpoints/cyclegan_bear_2/images/epoch150_real_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_2/images/epoch150_fake_B.png" width="40%"/>  
</td>
</tr>
<tr>
<td>
<img src="./checkpoints/cyclegan_bear_3/images/epoch150_fake_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_3/images/epoch150_real_B.png" width="40%"/>
</td>
<td>
<img src="./checkpoints/cyclegan_bear_3/images/epoch150_real_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_3/images/epoch150_fake_B.png" width="40%"/>  
</td>
</tr>

#### Epoch 200

<table border=1>
<tr>
<td>
<img src="./checkpoints/cyclegan_bear_2/images/epoch200_fake_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_2/images/epoch200_real_B.png" width="40%"/>
</td>
<td>
<img src="./checkpoints/cyclegan_bear_2/images/epoch200_real_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_2/images/epoch200_fake_B.png" width="40%"/>  
</td>
</tr>
<tr>
<td>
<img src="./checkpoints/cyclegan_bear_3/images/epoch200_fake_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_3/images/epoch200_real_B.png" width="40%"/>
</td>
<td>
<img src="./checkpoints/cyclegan_bear_3/images/epoch200_real_A.png" width="40%"/>
<img src="./checkpoints/cyclegan_bear_3/images/epoch200_fake_B.png" width="40%"/>  
</td>
</tr>

#### uncompleted Bi-cycle GAN


### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

### Others

### Reference
[1] Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks 
Jun-Yan Zhu∗ Taesung Park∗ Phillip Isola Alexei A. Efros, Berkeley AI Research (BAIR) laboratory, UC Berkeley
https://arxiv.org/pdf/1703.10593.pdf

[2] animals with attributes 2 A free dataset for Attribute Based Classification and Zero-Shot Learning
Christoph H. Lampert, Daniel Pucher, Johannes Dostal
https://cvml.ist.ac.at/AwA/
