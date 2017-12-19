# Homework4 report

### What scenario do I apply in?
I use the built-in data set, horse2zebra, as my first pair of domain transformation. As another pair of domain, I use the same horse dataset plus another giraffe dataset I downloaded online.

### Qualitative results
| Domain | Real A | Fake B | Real B | Fake A |
| :----: | :----: | :----: | :----: | :----: |
| A ←→ B | ![](horse2giraffe_results/image/epoch023_real_A.png) | ![](horse2giraffe_results/image/epoch023_fake_B.png) | ![](horse2giraffe_results/image/epoch023_real_B.png) | ![](horse2giraffe_results/image/epoch023_fake_A.png) |
| A ←→ B | ![](horse2giraffe_results/image/epoch026_real_A.png) | ![](horse2giraffe_results/image/epoch026_fake_B.png) | ![](horse2giraffe_results/image/epoch026_real_B.png) | ![](horse2giraffe_results/image/epoch026_fake_A.png) |
| A ←→ B | ![](horse2giraffe_results/image/epoch031_real_A.png) | ![](horse2giraffe_results/image/epoch031_fake_B.png) | ![](horse2giraffe_results/image/epoch031_real_B.png) | ![](horse2giraffe_results/image/epoch031_fake_A.png) |
| A ←→ B | ![](horse2giraffe_results/image/epoch034_real_A.png) | ![](horse2giraffe_results/image/epoch034_fake_B.png) | ![](horse2giraffe_results/image/epoch034_real_B.png) | ![](horse2giraffe_results/image/epoch034_fake_A.png) |
| A ←→ B | ![](horse2giraffe_results/image/epoch041_real_A.png) | ![](horse2giraffe_results/image/epoch041_fake_B.png) | ![](horse2giraffe_results/image/epoch041_real_B.png) | ![](horse2giraffe_results/image/epoch041_fake_A.png) |
| A ←→ B | ![](horse2giraffe_results/image/epoch051_real_A.png) | ![](horse2giraffe_results/image/epoch051_fake_B.png) | ![](horse2giraffe_results/image/epoch051_real_B.png) | ![](horse2giraffe_results/image/epoch051_fake_A.png) |
| A ←→ B | ![](horse2giraffe_results/image/epoch067_real_A.png) | ![](horse2giraffe_results/image/epoch067_fake_B.png) | ![](horse2giraffe_results/image/epoch067_real_B.png) | ![](horse2giraffe_results/image/epoch067_fake_A.png) |
| B ←→ C | ![](horse2zebra_results/image/epoch052_real_A.png) | ![](horse2zebra_results/image/epoch052_fake_B.png) | ![](horse2zebra_results/image/epoch052_real_B.png) | ![](horse2zebra_results/image/epoch052_fake_A.png) |
| B ←→ C | ![](horse2zebra_results/image/epoch053_real_A.png) | ![](horse2zebra_results/image/epoch053_fake_B.png) | ![](horse2zebra_results/image/epoch053_real_B.png) | ![](horse2zebra_results/image/epoch053_fake_A.png) |
| B ←→ C | ![](horse2zebra_results/image/epoch059_real_A.png) | ![](horse2zebra_results/image/epoch059_fake_B.png) | ![](horse2zebra_results/image/epoch059_real_B.png) | ![](horse2zebra_results/image/epoch059_fake_A.png) |
| B ←→ C | ![](horse2zebra_results/image/epoch115_real_A.png) | ![](horse2zebra_results/image/epoch115_fake_B.png) | ![](horse2zebra_results/image/epoch115_real_B.png) | ![](horse2zebra_results/image/epoch115_fake_A.png) |
| B ←→ C | ![](horse2zebra_results/image/epoch124_real_A.png) | ![](horse2zebra_results/image/epoch124_fake_B.png) | ![](horse2zebra_results/image/epoch124_real_B.png) | ![](horse2zebra_results/image/epoch124_fake_A.png) |
| B ←→ C | ![](horse2zebra_results/image/epoch136_real_A.png) | ![](horse2zebra_results/image/epoch136_fake_B.png) | ![](horse2zebra_results/image/epoch136_real_B.png) | ![](horse2zebra_results/image/epoch136_fake_A.png) |
| B ←→ C | ![](horse2zebra_results/image/epoch186_real_A.png) | ![](horse2zebra_results/image/epoch186_fake_B.png) | ![](horse2zebra_results/image/epoch186_real_B.png) | ![](horse2zebra_results/image/epoch186_fake_A.png) |
| B ←→ C | ![](horse2zebra_results/image/epoch195_real_A.png) | ![](horse2zebra_results/image/epoch195_fake_B.png) | ![](horse2zebra_results/image/epoch195_real_B.png) | ![](horse2zebra_results/image/epoch195_fake_A.png) |

### My thoughts
As the results shown, its easier for cycle gan to trainsform domains with weak pattern (horse) into domains with stronger pattern (zebra and giraffe), but not the other way around. It fails to transforms shapes attributes, such as long neck of giraffes. The model tends to generate giraffe without the neck while transforming back to horse, but it can't generate realistic horse with a short neck.

### Others
I uploaded the training files and image results in both horse2zebra_results and horse2giraffe_results directories.

### Reference
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
