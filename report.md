# Homework4 report

|apple|orange apple|
|-|-|
|<img src="https://i.imgur.com/EKls6PD.jpg" width="250">|<img src="https://imgur.com/IGZEMUs.jpg" width="250">|
||披著橘子皮的蘋果|

### What scenario do I apply in?
**Fruit peel transfer (果皮轉換訓練)**<br/>
data set 是 蘋果A、橘子B、香蕉C<br/>

思路: <br/>
蘋果(紅) ⇔ 橘子(橘) ⇔ 香蕉(黃) <br/>
這三個顏色是有漸層的，應該可以有著有意思的效果。 <br/>

### What do I modify?  <br/>
大膽嘗試使用不同與前者形狀不同的香蕉data set， <br/>
希望能訓練到香蕉的外皮，而不是香蕉的形狀。 <br/>
 <br/>
在code的部分，並未做太大的更動，遵循原作者的原始碼。([code](https://github.com/vanhuyz/CycleGAN-TensorFlow/blob/master/train.py)) <br/>
 <br/>
此外，參考資料裡面有蘋果跟橘子的data set，沒有香蕉的data set， <br/>
於是，我在google圖片、pixabay等網站爬了600張左右的香蕉照片來訓練。 <br/>

### Qualitative results <br/>

|apple|orange|banana|
|-|-|-|
|![](https://i.imgur.com/nawjiyS.jpg)|![](https://i.imgur.com/ozofolW.jpg)|![](https://i.imgur.com/xyT15M6.jpg)|
|![](https://i.imgur.com/AoJbOw7.jpg)|![](https://i.imgur.com/q8ludkP.jpg)|![](https://i.imgur.com/5HtAdli.jpg)|

### My experiment & thoughts <br/>

* 橘子跟香蕉的轉換(B→C)，並沒有生成出很好的圖片， <br/>
我認為有幾個原因： <br/>
    1. 香蕉的外型與前兩者有段落差 (蘋果跟橘子都是圓的，香蕉是長條狀的) 
    2. 訓練時間應該還不太足夠 (本實驗使用1080Ti，訓練3天) 
* 蘋果(A.) ⇔ 橘子(B.) <br/>
使用pretrain model (Reference[2]) <br/>
* 橘子(B.) ⇔ 香蕉(C.) <br/>
自行訓練 (也有著一些有趣的實驗照片) <br/>
    1. 香蕉2橘子，有時會很像木瓜(中間那張) <br/>
    ![](https://i.imgur.com/elCHoUi.png) <br/>
    2. 香蕉蒂頭的地方(紅色圈圈處)，也會被訓練進去， <br/>
    縮在橘子的某個區域， <br/>
    原本是希望他訓練黃色的果皮， <br/>
    沒預料到連蒂頭都訓練進去了。 <br/>
    ![](https://i.imgur.com/7IUAfKM.png) <br/>
    3. Experiment (cylcle loss) <br/>
    ![experiment cycle loss](https://i.imgur.com/SnaUDGr.png) <br/>
### Others (Discussion) <br/>
* data有時會有一些雜訊 <br/>
    * 比方說 有些照片旁邊有梨子 葡萄...等，不一定只有我想要的某種特定水果。 <br/>
* 篩選資料 感覺是最辛苦的一步 <br/>
    * 最辛苦的不是訓練的過程 (甚至是說，訓練的過程是輕鬆的，因為只是放著給他跑)。 <br/>
    * 收集資料的辛苦之處在於爬到的data，經常時會有各種奇形怪狀的圖片。 <br/>
    * 人工篩選掉不適當的圖片。 <br/>
    * 辛苦程度: 篩選資料(人工) > 收集資料(爬蟲) > 訓練過程 <br/>
* 看著實驗的進行 是有趣的 <br/>
### Reference <br/>
CycleGAN <br/>
* [CycleGAN](https://junyanz.github.io/CycleGAN) <br/>
* [CycleGAN-Tensorflow](https://github.com/vanhuyz/CycleGAN-TensorFlow) <br/>

Tool (image find): <br/>
* [Pixabay](https://pixabay.com)  <br/>
* [Download images from web page](https://imagecyborg.com)  <br/>
* [The Best Places to Find Free, High-Res Images for your Website](https://free.com.tw/15-find-free-images)  <br/>

Tool (image preprocess): <br/>
* [Scipy.imsave](https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imsave.html#scipy.misc.imsave) <br/>
* [Numpy.shuffle](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.shuffle.html) <br/>

Tool (others): <br/>
* [Imgur](https://imgur.com) <br/>
* [Pdf to markdown](http://pdf2md.morethan.io)  <br/>
* [Get image (word)](https://www.lhu.edu.tw/i/teach-online/word_pic.htm)  <br/>
* [Md change image size I](https://stackoverflow.com/questions/14675913/how-to-change-image-size-markdown) <br/>
* [Md change image size II](https://stackoverflow.com/questions/14675913/how-to-change-image-size-markdown) <br/>
* [Warning](https://github.com/tensorflow/tensorflow/issues/7778) <br/>
