# Homework4 report
高穎 106062525

### What scenario do I apply in?

Cycle GAN 可以做到兩個domain之間的轉換,這次實作的Bi-Cycle GAN可以做到三個domain之間的轉換

A->B 將衛星地圖轉換成google地圖

A->C 將衛星地圖轉化成圖畫的風格

### What do I modify? 
you can show some snippet

在原本Cycle GAN的基礎上,加入第三個Domain的資訊.包含dataset的路徑,A到C的generator,以及C到A的generator,以及discriminator,lossfunction等等

        
        # GAN loss D_A(G_AC(A))
        fake_C = self.netG_AC(self.real_A)
        pred_fake = self.netD_AC(fake_C)
        loss_G_AC = self.criterionGAN(pred_fake, True)
        
        # Forward cycle loss
        rec_AC = self.netG_C(fake_C)
        loss_cycle_AC = self.criterionCycle(rec_AC, self.real_A) * lambda_AC
        
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_G_AC + loss_G_C +          
                 loss_cycle_AC + loss_cycle_C + loss_idt_AC + loss_idt_C

### Qualitative results

Cycle GAN : A->B

| Domain | real A | fake B | real B |
| :----: | :-----:| :----: | :----: |
| A->B |![](data/data/real_AB1.png)| ![](data/data/fake_BA1.png) | ![](data/data/real_BA1.png) | ![](data/data/fake_AB1.png) |
| A->B |![](data/data/real_AB2.png)| ![](data/data/fake_BA2.png) | ![](data/data/real_BA2.png) | ![](data/data/fake_AB2.png) |

Cycle GAN : A->C

| Domain | real A | fake C | real C | fake A |
| :----: | :-----:| :----: | :----: | :----: | 
| A->C |![](data/data/real_AC1.png)| ![](data/data/fake_CA1.png) | ![](data/data/real_CA1.png) | ![](data/data/fake_AC1.png) |
| A->C |![](data/data/real_AC2.png)| ![](data/data/fake_CA2.png) | ![](data/data/real_CA2.png) | ![](data/data/fake_AC2.png) |

Bi-Cycle GAN : A->B and A->C

| Domain | real A | fake B | real B | fake A | real A | fake C | real C | fake A |
| :----: | :-----:| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| A->B and A->C |![](data/data/epoch093_real_A.png)| ![](data/data/epoch093_fake_B.png) | ![](data/data/epoch093_real_B.png) | ![](data/data/epoch093_fake_A.png) |![](data/data/epoch093_real_AC.png)| ![](data/data/epoch093_fake_C.png) | ![](data/data/epoch093_real_C.png) | ![](data/data/epoch093_fake_CA.png) |
| A->B and A->C |![](data/data/epoch095_real_A.png)| ![](data/data/epoch095_fake_B.png) | ![](data/data/epoch095_real_B.png) | ![](data/data/epoch095_fake_A.png) |![](data/data/epoch095_real_AC.png)| ![](data/data/epoch095_fake_C.png) | ![](data/data/epoch095_real_C.png) | ![](data/data/epoch095_fake_CA.png) |

### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

### Others

### Reference
