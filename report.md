# Homework4 report

### What scenario do I apply in?

利用CycleGAN處理unpaired image-to-image translation的問題。原本CycleGAN的方法目標是轉換`A``B`兩個domain的image，目標是先學習`G：A → B`，使得 G（A）生成出來的image的distribution與使用adversarial loss的distribution `B`無法區分，接著利用反向mapping`F : B → A`和cycle consistency loss讓 F(G(A)) 與原本`A`越相似。

本次作業目標實作BicycleGAN，達到cycles: `A -> B' -> A'` 和 `B -> C' -> B'`

`A`為summer Yosemite photos `B`為winter Yosemite photos `C`為Van Gogh paintings

Bi-CycleGAN達到同時轉換季節還可以轉換照片風格，增加照片風格多樣性，也可以運用在data augmentation。


### What do I modify? 

先分別train兩個CycleGANs，一個將夏季影像轉換成冬季，另一個將冬季影像轉換成Van Gogh圖畫風格。接著修改CycleGAN的loss function，將兩個Cycles`A <-> B` 和`B <-> C`一起訓練。

loss function
```

```

### Qualitative results
CycleGANs
<p><img src="summer2winter.jpg" width=20% /></p>
<p><img src="winter2vangogh.jpg" width=20% /></p>
Bi-CycleGAN
<p><img src="summer2winter.jpg" width=20% /></p>

### My thoughts 
you can make some comments on the your own homework, e.g. what's the strength? what's the limitation?

### Others

### Reference
J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In ICCV, 2017.

J.-Y. Zhu, R. Zhang, D. Pathak, T. Darrell, A. A. Efros, O. Wang, and E. Shechtman. Toward multimodal image-to-image translation. In Advances in Neural Information Processing Systems (NIPS), 2017.

