# Homework4 report

### What scenario do I apply in?  
利用 mnist 和負片 mnist 去處理 svhn的截圖數字資料  
在 mnist的高判斷比率狀況下 了解機器對 svhn的理解情形  

### What do I modify?   
1、2資料集為原作者的處理情況
A->B*->A 和 B->A*->B*流程   
2是讓mnist負片後代入原訓練過程的結果  
3是原先照著題目講到的  
A->B*->A 和 B->C*->B*流程  
但後續發現對B的擴展仍然只有如1、2的效果  
故補齊了  
B->A*->B 和 C->B*->C*流程  
形成最後的結果

### Qualitative results
model 1:  
![md1]()
Step [20000/20000], d_real_loss: 0.0447, d_mnist_loss: 0.0035, d_svhn_loss: 0.0412, d_fake_loss: 0.0404, g_loss: 1.1687  

model 2:  
![md2]()  
Step [20000/20000], d_real_loss: 0.0422, d_mnist_loss: 0.0020, d_svhn_loss: 0.0402, d_fake_loss: 0.0403, g_loss: 1.1395  

model 3:  
![md3]()
Step [20000/20000], d_real_loss: 0.0863, d_mnist_loss: 0.0056, d_svhn_loss: 0.0807, d_fake_loss: 0.2429, g_loss: 0.9839  

model 4:  
![md4_1]() ![md4_2]()    
Step [10000/10000], d_real_loss: 0.1051, d_mnist_loss: 0.0320, d_svhn_loss: 0.0731, d_fake_loss: 0.0835, g_loss: 1.1164  

### My thoughts 
在背景相似度高的時候生成mnist會有良好的效果
而生成svhn的時候由於mnist資料的單調情形 會生成比較單一的結果
後續在確定如何鎖定label之後或許可生成除了像數字之外 包含正確性的結果

### Others
純屬意外的狀況  
![mdout]()

### Reference
[原作者](https://github.com/yunjey/mnist-svhn-transfer)  