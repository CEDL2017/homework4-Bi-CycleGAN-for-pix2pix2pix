# Homework4 report

### What scenario do I apply in?
>X: horse  
>Y: zebre  
>Z: lion  

### What do I modify? 
#### Separate training
I simply train two seperate CycleGAN using this [Tensorflow implementation](https://github.com/vanhuyz/CycleGAN-TensorFlow) and download the images of lion from ImageNet dataset. 

#### Jointly training
I try to modified the code based on the CycleGAN above, and first thing is to define two cycle in the model. Then we define loss to improve the performance by adding bi-direction loss: `X<->Y & Y<->Z`, which shared Y domain.

### Qualitative results
#### horse <-> zebra
<table border=1>
<tr>
<td colspan="2">
horse → zebra
</td>
<td colspan="2">
zebra → horse
</td>
</tr>

<tr>
<td>
real horse
</td>
<td>
fake zebra
</td>
<td>
real zebra
</td>
<td>
fake horse
</td>
</tr>

<tr>
<td>
<img src="imgs/horse2zebra/real_horse1.jpg"/>
</td>
<td>
<img src="imgs/horse2zebra/fake_zebra1.jpg"/>
</td>
<td>
<img src="imgs/zebra2horse/real_zebra1.jpg"/>
</td>
<td>
<img src="imgs/zebra2horse/fake_horse1.jpg"/>
</td>
</tr>

<tr>
<td>
<img src="imgs/horse2zebra/real_horse2.jpg"/>
</td>
<td>
<img src="imgs/horse2zebra/fake_zebra2.jpg"/>
</td>
<td>
<img src="imgs/zebra2horse/real_zebra2.jpg"/>
</td>
<td>
<img src="imgs/zebra2horse/fake_horse2.jpg"/>
</td>
</tr>

<tr>
<td>
<img src="imgs/horse2zebra/real_horse3.jpg"/>
</td>
<td>
<img src="imgs/horse2zebra/fake_zebra3.jpg"/>
</td>
<td>
<img src="imgs/zebra2horse/real_zebra3.jpg"/>
</td>
<td>
<img src="imgs/zebra2horse/fake_horse3.jpg"/>
</td>
</tr>
</table>

#### zebra <-> lion
<table border=1>
<tr>
<td colspan="2">
zebra → lion
</td>
<td colspan="2">
lion → zebra
</td>
</tr>

<tr>
<td>
real zebra
</td>
<td>
fake lion
</td>
<td>
real zebra
</td>
<td>
fake horse
</td>
</tr>

<tr>
<td>
<img src="imgs/zebra2lion/real_zebra1.jpg"/>
</td>
<td>
<img src="imgs/zebra2lion/fake_lion1.jpg"/>
</td>
<td>
<img src="imgs/lion2zebra/lion1.jpg"/>
</td>
<td>
<img src="imgs/lion2zebra/zebra1.jpg"/>
</td>
</tr>

<tr>
<td>
<img src="imgs/zebra2lion/real_zebra2.jpg"/>
</td>
<td>
<img src="imgs/zebra2lion/fake_lion2.jpg"/>
</td>
<td>
<img src="imgs/lion2zebra/lion2.jpg"/>
</td>
<td>
<img src="imgs/lion2zebra/zebra2.jpg"/>
</td>
</tr>

<tr>
<td>
<img src="imgs/zebra2lion/real_zebra3.jpg"/>
</td>
<td>
<img src="imgs/zebra2lion/fake_lion3.jpg"/>
</td>
<td>
<img src="imgs/lion2zebra/lion3.jpg"/>
</td>
<td>
<img src="imgs/lion2zebra/zebra3.jpg"/>
</td>
</tr>
</table>

### My thoughts 
In this homework, I think I am confused at the BiCycleGAN TA and teacher want and the other one BicycleGAN that introduce in NIPs 2017. I think these two are totally different since one is based on CycleGAN and the other one is based on conditional GANs. What we are doing in this homework is much more similar to [starGAN](https://github.com/yunjey/StarGAN) from my perspective.
### Others

### Reference

