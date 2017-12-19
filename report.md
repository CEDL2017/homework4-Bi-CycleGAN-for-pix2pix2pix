# Homework4 report
高穎 106062525

### What scenario do I apply in?

Cycle GAN 可以做到兩個domain之間的轉換,這次實作的Bi-Cycle GAN可以做到三個domain之間的轉換

A->B 將衛星地圖轉換成google地圖

A->C 將衛星地圖轉化成圖畫的風格

### What do I modify? 
you can show some snippet

        
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
| Category name | Accuracy | Sample training images | Sample true positives | False positives with true label | False negatives with wrong predicted label |
| :-----------: | :-------:| :--------------------: | :-------------------: | :-----------------------------: | :----------------------------------------: |
| Kitchen |0.65| ![](thumbnails/Kitchen_train_image_0001.jpg) | ![](thumbnails/Kitchen_TP_image_0192.jpg) | ![](thumbnails/Kitchen_FP_image_0107.jpg) | ![](thumbnails/Kitchen_FN_image_0183.jpg) |
| Store |0.54| ![](thumbnails/Store_train_image_0001.jpg) | ![](thumbnails/Store_TP_image_0150.jpg) | ![](thumbnails/Store_FP_image_0026.jpg) | ![](thumbnails/Store_FN_image_0151.jpg) |
| Bedroom |0.49| ![](thumbnails/Bedroom_train_image_0001.jpg) | ![](thumbnails/Bedroom_TP_image_0180.jpg) | ![](thumbnails/Bedroom_FP_image_0121.jpg) | ![](thumbnails/Bedroom_FN_image_0176.jpg) |
| LivingRoom |0.39| ![](thumbnails/LivingRoom_train_image_0001.jpg) | ![](thumbnails/LivingRoom_TP_image_0145.jpg) | ![](thumbnails/LivingRoom_FP_image_0047.jpg) | ![](thumbnails/LivingRoom_FN_image_0147.jpg) |
| Office |0.94| ![](thumbnails/Office_train_image_0002.jpg) | ![](thumbnails/Office_TP_image_0185.jpg) | ![](thumbnails/Office_FP_image_0005.jpg) | ![](thumbnails/Office_FN_image_0144.jpg) |
| Industrial |0.57| ![](thumbnails/Industrial_train_image_0002.jpg) | ![](thumbnails/Industrial_TP_image_0152.jpg) | ![](thumbnails/Industrial_FP_image_0001.jpg) | ![](thumbnails/Industrial_FN_image_0148.jpg) |
| Suburb |0.99| ![](thumbnails/Suburb_train_image_0002.jpg) | ![](thumbnails/Suburb_TP_image_0176.jpg) | ![](thumbnails/Suburb_FP_image_0081.jpg) | ![](thumbnails/Suburb_FN_image_0013.jpg) |
| InsideCity |0.62| ![](thumbnails/InsideCity_train_image_0005.jpg) | ![](thumbnails/InsideCity_TP_image_0134.jpg) | ![](thumbnails/InsideCity_FP_image_0035.jpg) | ![](thumbnails/InsideCity_FN_image_0140.jpg) |
| TallBuilding |0.72| ![](thumbnails/TallBuilding_train_image_0010.jpg) | ![](thumbnails/TallBuilding_TP_image_0129.jpg) | ![](thumbnails/TallBuilding_FP_image_0059.jpg) | ![](thumbnails/TallBuilding_FN_image_0131.jpg) |
| Street |0.7| ![](thumbnails/Street_train_image_0001.jpg) | ![](thumbnails/Street_TP_image_0147.jpg) | ![](thumbnails/Street_FP_image_0128.jpg) | ![](thumbnails/Street_FN_image_0149.jpg) |
| Highway |0.81| ![](thumbnails/Highway_train_image_0009.jpg) | ![](thumbnails/Highway_TP_image_0162.jpg) | ![](thumbnails/Highway_FP_image_0079.jpg) | ![](thumbnails/Highway_FN_image_0144.jpg) |
| OpenCountry |0.51| ![](thumbnails/OpenCountry_train_image_0003.jpg) | ![](thumbnails/OpenCountry_TP_image_0125.jpg) | ![](thumbnails/OpenCountry_FP_image_0082.jpg) | ![](thumbnails/OpenCountry_FN_image_0123.jpg) |
| Coast |0.82| ![](thumbnails/Coast_train_image_0006.jpg) | ![](thumbnails/Coast_TP_image_0130.jpg) | ![](thumbnails/Coast_FP_image_0123.jpg) | ![](thumbnails/Coast_FN_image_0122.jpg) |
| Mountain |0.87| ![](thumbnails/Mountain_train_image_0002.jpg) | ![](thumbnails/Mountain_TP_image_0123.jpg) | ![](thumbnails/Mountain_FP_image_0124.jpg) | ![](thumbnails/Mountain_FN_image_0101.jpg) |
| Forest |0.93| ![](thumbnails/Forest_train_image_0003.jpg) | ![](thumbnails/Forest_TP_image_0142.jpg) | ![](thumbnails/Forest_FP_image_0101.jpg) | ![](thumbnails/Forest_FN_image_0128.jpg) |

### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

### Others

### Reference
