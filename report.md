# Homework4 report

### What scenario do I apply in?
I have tried on several scenarios including 
 1. baby -> alien and baby -> ape
 

 2. snake -> rope and snake -> earthworm

However, the results are not appealing. So I collected another dataset which is more friendly to train in my point of view.

 3. oat -> rice and oat-> raisin

### What do I modify? 
I didn't modify the cyclegan network. I only changed some parameters in the base_options and wrote matlab function to preprocess the downloaded pictures.

1. First, I use 'Bulk Image Downloader' to get pictures from google. However, the pictures are not always what you wanted, so you need to select them one by one. This ends up with only a few usable training datas (around 150 in single domain).

2. Secondly, I wrote a matlabe code to resize the pictures downloaded into 256 by 256 in a bulk and changed the title name at the same time.

3. Later, I changed the base_options parameter loadsize to 256 since my data had already resized. And I didn't crop the pictures.

4. Last, I run the cyclegan directly twice to get my results.


### Qualitative results


### My thoughts 
I think data collection plays an important role in cyclegan. If the background is too complicated, the network will be hard to transform. Also, the main objects between each domain should share a quite similar geometrical shape or simply the contours.
One more, the pictures within a single domain sholdn't vary too much (especially when the datasets are not big enough).
Finally, even if the datasets are small, with a refined data collection datasests, the results can still be rather good.

### Others

### Reference
