# Homework4 105061516 Report

### What scenario do I apply in?
A: cow  
B: buffalo    
C: rhinoceros   
Why: 預想是讓其他牛加上乳牛花紋、氂牛的毛、或者犀牛的角，進而可能在隨機輸入圖片產生效果。

### What do I modify? 
1. Train on subsets of Animals with Attributes 2
2. Add some Bi-CycleGAN implement  
  + Code/Cmodels/bicycle_gan_model.py     
  + Code/data/biunaligned_dataset.py

### Qualitative results
> put some interesting images generated from your Bi-CycleGANs
<table border=2 align=center  width="100%">
  做得不錯的CASE
<tr>
  <td>
    real_A
  </td>
  <td>
    bi AtoB
  </td>
  <td>
    AtoB
  </td>
  <td>
    bi AtoC
  </td>
  <td>
    AtoC
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10677_real_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10677_fake_BA.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10677_fake_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10677_fake_BC.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/cow_11113_fake_B.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10215_real_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10215_fake_BA.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10215_fake_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10215_fake_BC.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/cow_10307_fake_B.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    real_B
  </td>
  <td>
    bi BtoA
  </td>
  <td>
    BtoA
  </td>
  <td>
    bi BtoAtoC
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10834_real_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10834_fake_AB.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10834_fake_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10834_rec_ABC.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10090_real_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10090_fake_AB.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10090_fake_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10090_rec_ABC.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    real_C
  </td>
  <td>
    bi CtoA
  </td>
  <td>
    CtoA
  </td>
  <td>
    bi CtoAtoB
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/cow_10185_real_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10162_fake_CB.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/cow_10185_fake_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10162_rec_CBA.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10215_real_C.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10215_fake_CB.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/cow_10307_fake_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10215_rec_CBA.png" width="100%"/>
  </td>
</tr>
</table>
<table border=2 align=center  width="100%">
  做得很差的CVASE
<tr>
  <td>
    real_A
  </td>
  <td>
    bi AtoB
  </td>
  <td>
    AtoB
  </td>
  <td>
    bi AtoC
  </td>
  <td>
    AtoC
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10743_real_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10743_fake_BA.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10743_fake_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10743_fake_BC.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/cow_11149_fake_B.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10196_real_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10196_fake_BA.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10196_fake_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10196_fake_BC.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/cow_10305_fake_B.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    real_B
  </td>
  <td>
    bi BtoA
  </td>
  <td>
    BtoA
  </td>
  <td>
    bi BtoAtoC
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10754_real_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10754_fake_AB.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10754_fake_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10754_rec_ABC.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/buffalo_10594_real_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10594_fake_AB.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10594_fake_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10594_rec_ABC.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    real_C
  </td>
  <td>
    bi CtoA
  </td>
  <td>
    CtoA
  </td>
  <td>
    bi CtoAtoB
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/cow_10023_real_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10021_fake_CB.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/cow_10023_fake_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10021_rec_CBA.png" width="100%"/>
  </td>
</tr>
<tr>
  <td>
    <img src="report_img/cow_10047_real_B.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10046_fake_CB.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/cow_10047_fake_A.png" width="100%"/>
  </td>
  <td>
    <img src="report_img/buffalo_10046_rec_CBA.png" width="100%"/>
  </td>
</tr>
</table>

### My thoughts 
> you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?
<br>
由於task都是部份區域，而非整張圖片。
故背景過於雜亂，或前景物體單調到與背景合為一體的輸入，會導致輸出效能較差。
bi-cycle的loss讓GAN不會學到太偏，補足原本 noisy data 造成的嚴重影響
最終task效果與預期之間相差不少，歸因於下列:
1. 原始Data背景過於複雜，三類牛所在的環境截然不同，讓GAN學習多餘的部分(背景)
2. CycleGANs 本身偏向 Texture 變更，上述的動機中，也想要對 Edge 做變更，不適非常貼合預想的功能



### Others

### Reference
CycleGan: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Animals with Attributes 2: https://cvml.ist.ac.at/AwA2/

