# Homework4 report

### What scenario do I apply in?
在 Cycle GAN 的 paper，已經有 yosemite，夏天到冬天的轉換，我想根據這個work再增加轉換到秋天的功能。

Yosemite Autumn -> Summer -> Winter

<table>
  <tr>
    <td>Autumn</td>
    <td><img src="image/autumn_1.jpg"/></td>
    <td><img src="image/autumn_2.jpg"/></td>
    <td><img src="image/autumn_3.jpg"/></td>
    <td><img src="image/autumn_4.jpg"/></td>
  </tr>
  <tr>
    <td>Summer</td>
    <td><img src="image/summer_1.jpg"/></td>
    <td><img src="image/summer_2.jpg"/></td>
    <td><img src="image/summer_3.jpg"/></td>
    <td><img src="image/summer_4.jpg"/></td>
  </tr>
  <tr>
    <td>Winter</td>
    <td><img src="image/winter_1.jpg"/></td>
    <td><img src="image/winter_2.jpg"/></td>
    <td><img src="image/winter_3.jpg"/></td>
    <td><img src="image/winter_4.jpg"/></td>
  </tr>
</table>
### What do I modify? 

#### Data Collection

- 爬Google image和Flickr的圖片，參考image crawler https://github.com/hellock/icrawler
- 手動挑掉黑白或是不相關的圖片

#### Data Preprocess

- 把爬到的資料轉成256\*256，先把短邊resize到256，再center crop得到256\*256的圖


### Qualitative results
put some interesting images generated from your Bi-CycleGANs

#### Yosemite Autumn -> Summer -> Winter

<table>
  <tr>
    <td><img src="image/2011-06-14 232930_real_B.png"/></td>
    <td><img src="image/2011-06-14 232930_fake_A.png"/></td>
    <td><img src="image/2011-06-14 232930_fake_A_fake_B.png"/></td>
  </tr>
  <tr>
    <td><img src="image/2011-08-28 064410_real_B.png"/></td>
    <td><img src="image/2011-08-28 064410_fake_A.png"/></td>
    <td><img src="image/2011-08-28 064410_fake_A_fake_B.png"/></td>
  </tr>
  <tr>
    <td><img src="image/2011-08-30 231310_real_B.png"/></td>
    <td><img src="image/2011-08-30 231310_fake_A.png"/></td>
    <td><img src="image/2011-08-30 231310_fake_A_fake_B.png"/></td>
  </tr>
  <tr>
    <td><img src="image/2012-09-19 154901_real_B.png"/></td>
    <td><img src="image/2012-09-19 154901_fake_A.png"/></td>
    <td><img src="image/2012-09-19 154901_fake_A_fake_B.png"/></td>
  </tr>
  <tr>
    <td><img src="image/2013-05-30 195900_real_B.png"/></td>
    <td><img src="image/2013-05-30 195900_fake_A.png"/></td>
    <td><img src="image/2013-05-30 195900_fake_A_fake_B.png"/></td>
  </tr>
  
  <tr>
    <td><img src="image/2013-07-12 203251_real_B.png"/></td>
    <td><img src="image/2013-07-12 203251_fake_A.png"/></td>
    <td><img src="image/2013-07-12 203251_fake_A_fake_B.png"/></td>
  </tr>
  <tr>
    <td><img src="image/2014-07-19 174821_real_B.png"/></td>
    <td><img src="image/2014-07-19 174821_fake_A.png"/></td>
    <td><img src="image/2014-07-19 174821_fake_A_fake_B.png"/></td>
  </tr>
  <tr>
    <td><img src="image/2016-09-20 144731_real_B.png"/></td>
    <td><img src="image/2016-09-20 144731_fake_A.png"/></td>
    <td><img src="image/2016-09-20 144731_fake_A_fake_B.png"/></td>
  </tr>
  
</table>



### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

### Others

### Reference
