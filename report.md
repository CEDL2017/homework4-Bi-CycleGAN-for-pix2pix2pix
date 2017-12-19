# Homework4 report

### What scenario do I apply in?

Domain A: Winter landscape Dataset
Domain B: Summer landscape Dataset
Domain C: Flowers Dataset

I trained A<-->B and B<-->C

### What do I modify? 


### Qualitative results
| Model name | Real A | Real B | Fake B | Fake A |
| :--------: | :----: | :----: | :----: | :----: |
|   A <-> B  | ![](sum2win/cherry_pick/epoch189_real_A.png) | ![](sum2win/cherry_pick/epoch189_real_B.png) | ![](sum2win/cherry_pick/epoch189_fake_B.png) | ![](sum2win/cherry_pick/epoch189_fake_A.png) |
|   A <-> B  | ![](sum2win/cherry_pick/epoch190_real_A.png) | ![](sum2win/cherry_pick/epoch190_real_B.png) | ![](sum2win/cherry_pick/epoch190_fake_B.png) | ![](sum2win/cherry_pick/epoch190_fake_A.png) |
|   A <-> B  | ![](sum2win/cherry_pick/epoch194_real_A.png) | ![](sum2win/cherry_pick/epoch194_real_B.png) | ![](sum2win/cherry_pick/epoch194_fake_B.png) | ![](sum2win/cherry_pick/epoch194_fake_A.png) |
|   A <-> B  | ![](sum2win/cherry_pick/epoch196_real_A.png) | ![](sum2win/cherry_pick/epoch196_real_B.png) | ![](sum2win/cherry_pick/epoch196_fake_B.png) | ![](sum2win/cherry_pick/epoch196_fake_A.png) |
|   B <-> C  | ![](sum2flower/cherry_pick/epoch054_real_A.png) | ![](sum2flower/cherry_pick/epoch054_real_B.png) | ![](sum2flower/cherry_pick/epoch054_fake_B.png) | ![](sum2flower/cherry_pick/epoch054_fake_A.png) |
|   B <-> C  | ![](sum2flower/cherry_pick/epoch057_real_A.png) | ![](sum2flower/cherry_pick/epoch057_real_B.png) | ![](sum2flower/cherry_pick/epoch057_fake_B.png) | ![](sum2flower/cherry_pick/epoch057_fake_A.png) |
|   B <-> C  | ![](sum2flower/cherry_pick/epoch067_real_A.png) | ![](sum2flower/cherry_pick/epoch067_real_B.png) | ![](sum2flower/cherry_pick/epoch067_fake_B.png) | ![](sum2flower/cherry_pick/epoch067_fake_A.png) |

### My thoughts 
1. The transformation between winter and summer work quite good. When doing A->B, we can see that snow in the mountain melted and grass grows. When it's B->A, we can see some snow landscape appears.

2. The distance between domain B and C are pretty large, so the network can't learn too well.
It sometimes tried to turn whole landscape into multiple flowers instead of plant some flowers in the landscape.
But when it comes to C->B, it still works well to eliminate flowers in Domain C.

### Reference
junyanz's Pytorch implementation: [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
junyanz's [project page](https://junyanz.github.io/CycleGAN/)

