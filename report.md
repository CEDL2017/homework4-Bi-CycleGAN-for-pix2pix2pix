# Homework4 report

### What scenario do I apply in?
本次實驗是想將 B:猩猩 分別轉換成 A:外星人和 C:人類，來看看外星人有沒有可能也是從猩猩演化而來的。
嘗試了兩種不同的方法
#### Cycle-gan
使用 https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 的code來產生結果。
#### Bicycle-gan
根據 Cycle-gan 再自己另外加了另一個額外的循環，主要添加的循環程式碼如下。
### In bicycle-gan what do I modify? 
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
#### Cycle-gan
外星人-猩猩
<td><img src="4.png" width=900 height=450></td>
人類-猩猩
<td><img src="5.png" width=900 height=450></td>

#### Bicycle-gan
<td><img src="2017-12-15 13-23-35 的螢幕擷圖.png" width=900 height=450></td>
<td><img src="1.png" width=900 height=450></td>
<td><img src="2.png" width=900 height=450></td>

### My thoughts 
參考 cycle-gan 之後自己做了code上的修改變成 Bicycle-gan並經過200 epoch training，上面為部分的結果圖(有些圖下面的下標有錯，如:fake_BfromA <--> fake_AfromB, fake_BfromC <--> fake_CfromB)
        
        
雖然有些圖表現得還不錯，不過由實驗的結果我認為 Cycle-gan or Bicycle-gan 似乎比較是針對紋理色彩上的轉換，而不是真正的把一張猩猩的臉變成人臉(或外星人的臉變成猩猩的臉，或許也有可能是我訓練的epoch數不夠多。   

### Others
以上所有訓練資料都是從 google圖片上面經過篩選後抓下來的，
猩猩 815張，人類 554張，外星人 250張
### Reference
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
