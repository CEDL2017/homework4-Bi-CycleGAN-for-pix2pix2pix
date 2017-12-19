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

- 從以上sample可以看出，3個季節主要的差別從上圖來看是顏色，秋天橘紅色較多，夏天綠色的樹木較多，冬天多為雪白色。
- 希望他能學會顏色的轉換，更好的是可以轉換成冬天時，除了變白色，也把雪的texture學起來

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

- 顏色方面轉換的很好，樹葉從橘紅轉綠轉深，草地從綠轉白
- 雪地的texture也學的蠻像的

### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

1. 從上圖來看，cycle gan在texture的轉換可以學的不錯。
2. 可是圖片的細節還是有不少瑕疵可以辨認出是假的圖片。
3. 我認為cycle gan的只學到哪些對應的texture做轉換後，可以讓discriminator分不出真假。而不像pixel2pixel 是真的學到semantic的資訊，所以可以做更複雜的轉換，例如：image to segmentation label or 衛星圖轉成地圖。因此cycle gan的應用還是比較限制的。


### Reference
 https://github.com/hellock/icrawler
