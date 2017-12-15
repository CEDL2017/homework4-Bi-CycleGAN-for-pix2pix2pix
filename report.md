# Homework4 report

### What scenario do I apply in?
A : Alien B : 
### What do I modify? 
#### Discriminator of C
```
def backward_D_C(self):
        fake_C = self.fake_BtoC_pool.query(self.fake_C)
        loss_D_C = self.backward_D_basic(self.netD_C, self.real_C, fake_C)
        self.loss_D_C = loss_D_C.data[0]
```
#### Loss of recurrence C and recurrence BfromC
```
rec_C = self.netG_BtoC(fake_BfromC)
loss_cycle_C = self.criterionCycle(rec_C, self.real_C) * lambda_C
rec_BfromC = self.netG_C(fake_C)
loss_cycle_BfromC = self.criterionCycle(rec_BfromC, self.real_B) * lambda_B
fake_C = self.netG_BtoC(self.real_B)
pred_fake_BtoC = self.netD_C(fake_C)
```  

### Qualitative results
<td bgcolor=LightBlue><img src="thumbnails/Kitchen_image_0208.jpg" width=100 height=75></td>
<td bgcolor=LightBlue><img src="thumbnails/Kitchen_image_0074.jpg" width=107 height=75></td>
<td bgcolor=LightGreen><img src="thumbnails/Kitchen_image_0048.jpg" width=100 height=75></td>
### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

### Others

### Reference
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
