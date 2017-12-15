# Homework4 report

### What scenario do I apply in?
本次實驗是想將 B:猩猩 分別轉換成 A:外星人和 C:人類
嘗試了兩種不同的方法
#### Cycle-gan
使用 
#### Bicycle-gan
根據 Cycle-gan 再另外做些許修改
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
<td><img src="2017-12-15 13-23-35 的螢幕擷圖.png" width=900 height=450></td>
<td><img src="1.png" width=900 height=450></td>
<td><img src="2.png" width=900 height=450></td>

### My thoughts 
參考 cycle-gan 之後自己做了code上的修改變成 Bicycle-gan並經過200 epoch training，上面為部分的結果圖(有些圖下面的下標有些許錯，fake_BfromA <--> fake_AfromB, fake_BfromC <--> fake_CfromB)
        
        
不過由實驗的結果也可以看出來雖然有些圖表現得還不錯，但是也有一些圖感覺只是單純地作色彩轉換而已，似乎並沒有取道有效的 feature。   

### Others

### Reference
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
