# Homework4 report

### What scenario do I apply in?

>Domain A: women pictures   
>Domain B: body part segmenation  
>Domain C: child pictures    

I trained A<->B and B<->C

### What do I modify? 

<code>python train.py --dataroot ./datasets/XXX --name XXX_cyclegan --model cycle_gan</code>

### Qualitative results
| Model name | Real A | Fake B | Real B | Fake A |
| :--------: | :----: | :----: | :----: | :----: |
| A<->B | ![](sum2win/cherry_pick/epoch189_real_A.png) | ![](sum2win/cherry_pick/epoch189_real_B.png) | ![](sum2win/cherry_pick/epoch189_fake_B.png) | ![](sum2win/cherry_pick/epoch189_fake_A.png) |
| A<->B | ![](sum2win/cherry_pick/epoch190_real_A.png) | ![](sum2win/cherry_pick/epoch190_real_B.png) | ![](sum2win/cherry_pick/epoch190_fake_B.png) | ![](sum2win/cherry_pick/epoch190_fake_A.png) |
| B<->C | ![](sum2flower/cherry_pick/epoch054_real_A.png) | ![](sum2flower/cherry_pick/epoch054_real_B.png) | ![](sum2flower/cherry_pick/epoch054_fake_B.png) | ![](sum2flower/cherry_pick/epoch054_fake_A.png) |
| B<->C | ![](sum2flower/cherry_pick/epoch057_real_A.png) | ![](sum2flower/cherry_pick/epoch057_real_B.png) | ![](sum2flower/cherry_pick/epoch057_fake_B.png) | ![](sum2flower/cherry_pick/epoch057_fake_A.png) |

| Model name | Real A | Real B | Fake B | Fake A |
| :--------: | :----: | :----: | :----: | :----: |
| A<->B<->C | ![](sum2win/cherry_pick/epoch189_real_A.png) | ![](sum2win/cherry_pick/epoch189_real_B.png) | ![](sum2win/cherry_pick/epoch189_fake_B.png) | ![](sum2win/cherry_pick/epoch189_fake_A.png) |


### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

In the women <-> segmentation transformation, the body segmentation is roughly transformed to the corresponding body (skin, clothes, face). But the image background is noisy and meaningless. I think this is because the images contain only body part segmentaion while no background segmentaion, so learning a structured background is hard for the model. In the child <-> segmentation transformation, the transformation is and 

### Others

### Reference
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
