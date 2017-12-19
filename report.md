# Homework4 report

### What scenario do I apply in?
you are encouraged to elaborate the motivation here
>Domain A: women pictures   
>Domain B: body part segmenation  
>Domain C: child pictures    

I trained A<->B and B<->C

### What do I modify? 
you can show some snippet

### Qualitative results
| Model name | Real A | Real B | Fake B | Fake A |
| :--------: | :----: | :----: | :----: | :----: |
| A<->B | ![](sum2win/cherry_pick/epoch189_real_A.png) | ![](sum2win/cherry_pick/epoch189_real_B.png) | ![](sum2win/cherry_pick/epoch189_fake_B.png) | ![](sum2win/cherry_pick/epoch189_fake_A.png) |
| A<->B | ![](sum2win/cherry_pick/epoch190_real_A.png) | ![](sum2win/cherry_pick/epoch190_real_B.png) | ![](sum2win/cherry_pick/epoch190_fake_B.png) | ![](sum2win/cherry_pick/epoch190_fake_A.png) |
| A<->B | ![](sum2win/cherry_pick/epoch194_real_A.png) | ![](sum2win/cherry_pick/epoch194_real_B.png) | ![](sum2win/cherry_pick/epoch194_fake_B.png) | ![](sum2win/cherry_pick/epoch194_fake_A.png) |
| A<->B | ![](sum2win/cherry_pick/epoch196_real_A.png) | ![](sum2win/cherry_pick/epoch196_real_B.png) | ![](sum2win/cherry_pick/epoch196_fake_B.png) | ![](sum2win/cherry_pick/epoch196_fake_A.png) |
| B<->C | ![](sum2flower/cherry_pick/epoch054_real_A.png) | ![](sum2flower/cherry_pick/epoch054_real_B.png) | ![](sum2flower/cherry_pick/epoch054_fake_B.png) | ![](sum2flower/cherry_pick/epoch054_fake_A.png) |
| B<->C | ![](sum2flower/cherry_pick/epoch057_real_A.png) | ![](sum2flower/cherry_pick/epoch057_real_B.png) | ![](sum2flower/cherry_pick/epoch057_fake_B.png) | ![](sum2flower/cherry_pick/epoch057_fake_A.png) |
| B<->C | ![](sum2flower/cherry_pick/epoch067_real_A.png) | ![](sum2flower/cherry_pick/epoch067_real_B.png) | ![](sum2flower/cherry_pick/epoch067_fake_B.png) | ![](sum2flower/cherry_pick/epoch067_fake_A.png) |

### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?
1. women <-> segmentation
2. child <-> segmentation
  
### Others

### Reference
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
