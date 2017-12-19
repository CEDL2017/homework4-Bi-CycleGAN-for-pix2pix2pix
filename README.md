106062623 楊朝勛

# Homework4 Bi-CycleGAN for Image-to-Image-to-Image Translation

Overview
---
CycleGAN is designed for **Unpaired Image-to-Image Translation** and many interesting applications of CycleGAN can be found in the paper and on their project [website](https://junyanz.github.io/CycleGAN/). Here we want to make good use of it and extend it to multiple cycles between different modalities.

Motivation: with CycleGAN, we can better transfer the styles of images from domain A to domain B and vice versa without unpaired data. What will it be if we have two cycles (A, B) and (B, C) where B is the common modality shared by the two cycles? **Your task is to find some scenarios and related datasets that can demonstrate the idea and the advantages of Bi-CycleGAN for image-to-image-to-image transfer.** :fire:

For confused students, here are some concrete examples:

>Cycle 1: A ←→ B   
>Cycle 2: B ←→ C

B is the common domain (or common modality) shared by the two cycles.

Examples:   
>[Automatically generated dataset]   
>A: MNIST   
>B: Inverted MNIST (black->white, white->black)   
>C: Red-Green MNIST (black->red, white->green)   

Or

>A: An RGB image containing a person   
>B: Depth map   
>C: Keypoints (body joints)   

Or

>A: object captured from 0 degree    
>B: object captured from 30 degree   
>C: object captured from 60 degree   
>[Can be obtained by ShapeNet]

More other examples might be inspired by [here](https://github.com/mingyuliutw/UNIT)

Implementation
---
<br/>
    In the implementation of model, there're two dataset we have provided: s2f(spring to fall) and s2w(spring to winter). You can set the your own dataset in ./input to train your own model.
    
    
Installation
---
env setting:
if python 2.X use pip to install
if python 3.X use pip3 to install

using keras(tensorflow backend):
 ```
            pip install tensorflow
            pip install keras
            pip install scikit-image
            pip install matplotlib
            pip install Pillow
 ```


### Training            
Create the csv file as input to the data loader. 
	Edit the cyclegan_datasets.py file. For example, if you have a s2w dataset which contains 800 face images and 1000 ramen images both in PNG format, you can just edit the cyclegan_datasets.py as following:
	```python
	DATASET_TO_SIZES = {
    's2w_train': 1000
	}

	PATH_TO_CSV = {
    's2w_train': './input/s2w/s2w_train.csv'
	}

	DATASET_TO_IMAGETYPE = {
    'face2ramen_train': '.png'
	}

	``` 
	Run create_cyclegan_dataset.py:
	```bash
	python create_cyclegan_dataset.py --image_path_a=folder_a --image_path_b=folder_b --dataset_name="s2w_train" --do_shuffle=0
	```
Train the  model:
    if you want to train a whole new model remenber to remove `scorenet.h5` in model folder
    
    ```
    python main.py \
    --to_train=1 \
    --log_dir=./output/cyclegan/exp_01 \
    --config_filename=./configs/exp_01.json
    ```

### Restoring from the previous checkpoint:

    ```
    python main.py \
    --to_train=2 \
    --log_dir=./output/cyclegan/exp_01 \
    --config_filename=./configs/exp_01.json \
    --checkpoint_dir=./output/cyclegan/exp_01/#your_ckpt_file_name#
    ```

### Testing
Create the testing dataset.
	Edit the cyclegan_datasets.py file the same way as training.
	Create the csv file as the input to the data loader. 
	```bash
	python create_cyclegan_dataset.py --image_path_a=folder_a --image_path_b=folder_b --dataset_name="s2w_test" --do_shuffle=0
	```
Run testing.
```bash
    python main.py \
    --to_train=0 \
    --log_dir=./output/cyclegan/exp_01 \
    --config_filename=./configs/exp_01_test.json
    --checkpoint_dir=./output/cyclegan/exp_01/#your_ckpt_file_name#
```


### Results
---
一開始使用網路上夏天和冬天的dataset當作第一個dataset,
後來自己去蒐集秋天風景的圖片dataset,一開始在做training發現結果不盡理想,圖片會顏色很容易過度渲染(即使經過好幾個回合)
後來透過觀察dataset中的特性後,將自己所蒐集的fall_dataset中的圖片在進行挑選,使效果大幅提昇

下圖中：關於可看出圖片受到過度的渲染(第一張為結果,第二章為ground truth,training 77 epoch)


![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/fakeA_77_4.jpg)
![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/inputA_77_4.jpg)


最後經過挑選後的dataset(第一張為結果,第二章為ground truth,training 37 epoch)

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/fakeA_37_10.jpg)
![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/inputA_37_10.jpg)


* Summer to Winter

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2w/inputA_0_57.jpg)
![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2w/fakeA_0_57.jpg)

* Winter to Summer

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2w/inputB_0_261.jpg)
![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2w/fakeB_0_261.jpg)
  
* Summer to Fall

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/inputA_0_126.jpg)
![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/fakeA_0_126.jpg)

* Fall to Summer

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/inputB_0_19.jpg) 
![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/fakeB_0_19.jpg)
  
* Summer to Fall to Winter

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/inputA_0_3.jpg)

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/fakeA_0_3.jpg)

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2w/fakeA_0_3.jpg)

* Summer to Winter to Fall

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/inputA_0_50.jpg)

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2w/fakeA_0_50.jpg)

![](https://github.com/sun52525252/homework4-Bi-Cycle-GAN/blob/master/result/s2f/fakeA_0_50.jpg)
