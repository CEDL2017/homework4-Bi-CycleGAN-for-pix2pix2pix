# Homework4 report

### What scenario do I apply in?
I want to change human facial expressions, such as smile to sad.
<br>
**A: angry face   B: smile face   C: serious face**  
* Cycle 1: A <-> B    
* Cycle 2: B <-> C  

Dataset is from [FER2013](https://github.com/Microsoft/FERPlus)
### What do I modify? 
* Use Cycle-GAN from https://github.com/xhujoy/CycleGAN-tensorflow to train two seperate cycles.
* Bi-Cycle-GAN still working on  

### Qualitative results
* Cycle 1
  * good result  
      
    |real A|fake B|real B|fake A|
    |----|----|----|----|
    |<img src='./cycle_1_result/fer0035535.png'/>|<img src='./cycle_1_result/BtoA_fer0035535.png'/>|<img src='./cycle_1_result/fer0032230.png'/>|<img src='./cycle_1_result/AtoB_fer0032230.png'/>|
    |<img src='./cycle_1_result/fer0035625.png'/>|<img src='./cycle_1_result/BtoA_fer0035625.png'/>|<img src='./cycle_1_result/fer0032490.png'/>|<img src='./cycle_1_result/AtoB_fer0032490.png'/>|
 
  * fail result  
     
    |real A|fake B|real B|fake A|
    |----|----|----|----|
    |<img src='./cycle_1_result/fer0034668.png'/>|<img src='./cycle_1_result/BtoA_fer0034668.png'/>|<img src='./cycle_1_result/fer0032546.png'/>|<img src='./cycle_1_result/AtoB_fer0032546.png'/>|
    |<img src='./cycle_1_result/fer0032403.png'/>|<img src='./cycle_1_result/BtoA_fer0032403.png'/>|<img src='./cycle_1_result/fer0034313.png'/>|<img src='./cycle_1_result/AtoB_fer0034313.png'/>|
    
* Cycle 2
  * good result  
      
    |real C|fake B|real B|fake C|
    |----|----|----|----|
    |<img src='./cycle_2_result/fer0035663.png'/>|<img src='./cycle_2_result/AtoB_fer0035663.png'/>|<img src='./cycle_2_result/fer0032388.png'/>|<img src='./cycle_2_result/BtoA_fer0032388.png'/>|
    |<img src='./cycle_2_result/fer0035301.png'/>|<img src='./cycle_2_result/AtoB_fer0035301.png'/>|<img src='./cycle_2_result/fer0032900.png'/>|<img src='./cycle_2_result/BtoA_fer0032900.png'/>|
 
  * fail result  
     
    |real C|fake B|real B|fake |
    |----|----|----|----|
    |<img src='./cycle_2_result/fer0035632.png'/>|<img src='./cycle_2_result/AtoB_fer0035632.png'/>|<img src='./cycle_2_result/fer0032351.png'/>|<img src='./cycle_2_result/BtoA_fer0032351.png'/>|
    |<img src='./cycle_2_result/fer0032727.png'/>|<img src='./cycle_2_result/AtoB_fer0032727.png'/>|<img src='./cycle_2_result/fer0032891.png'/>|<img src='./cycle_2_result/BtoA_fer0032891.png'/>|

### My thoughts 
It is hard for using CycleGAN to do this task. Sometimes it just not change the facial expresssion, it also add something like bread, hair on original face. Maybe we can add some constraints or conditions to help training. Or, maybe Bi-Cycle GAN will do better.    

### Reference
* [CycleGAN](https://arxiv.org/abs/1703.10593)
* [FER2013](https://github.com/Microsoft/FERPlus)  
* [CycleGAN-tensorflow](https://github.com/xhujoy/CycleGAN-tensorflow)  
