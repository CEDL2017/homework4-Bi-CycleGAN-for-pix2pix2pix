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
| Domain | real A | fake B | real B | fake A | real A | fake C | real C | fake A |
| :----: | :-----:| :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| A->B |0.65| ![](thumbnails/data/data/realAB1.png) | ![](thumbnails/Kitchen_TP_image_0051.jpg) | ![](thumbnails/Kitchen_FP_image_0197.jpg) | ![](thumbnails/Kitchen_FN_image_0111.jpg) |
| A->B |0.54| ![](thumbnails/Store_train_image_0298.jpg) | ![](thumbnails/Store_TP_image_0099.jpg) | ![](thumbnails/Store_FP_image_0251.jpg) | ![](thumbnails/Store_FN_image_0090.jpg) |
| A->C |0.49| ![](thumbnails/Bedroom_train_image_0143.jpg) | ![](thumbnails/Bedroom_TP_image_0215.jpg) | ![](thumbnails/Bedroom_FP_image_0338.jpg) | ![](thumbnails/Bedroom_FN_image_0016.jpg) |
| A->C |0.94| ![](thumbnails/Office_train_image_0149.jpg) | ![](thumbnails/Office_TP_image_0183.jpg) | ![](thumbnails/Office_FP_image_0356.jpg) | ![](thumbnails/Office_FN_image_0127.jpg) |
| A->B->C |0.99| ![](thumbnails/Suburb_train_image_0157.jpg) | ![](thumbnails/Suburb_TP_image_0034.jpg) | ![](thumbnails/Suburb_FP_image_0180.jpg) | ![](thumbnails/Suburb_FN_image_0053.jpg) |
| A->B->C |0.62| ![](thumbnails/InsideCity_train_image_0143.jpg) | ![](thumbnails/InsideCity_TP_image_0060.jpg) | ![](thumbnails/InsideCity_FP_image_0029.jpg) | ![](thumbnails/InsideCity_FN_image_0084.jpg) |


### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

### Others

### Reference
