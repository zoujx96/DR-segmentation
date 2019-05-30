To train the model, run ```python train.py --seed 765 --preprocess '2' --lesion 'MA'``` for training a HEDNet model to segment Microaneurysm lesion images with preprocessing method of Contrast Enhancement using random seed 765.

The meaning of each preprocessing index is indicated in the following table.

| Preprocessing Index | Preprocessing Methods |
| :---: | :---: |
| '0' | None |
| '1' | Brightness Balance |
| '2' | Contrast Enhancement |
| '3' | Contrast Enhancement + Brightness Balance |
| '4' | Denoising |
| '5' | Denoising + Brightness Balance |
| '6' | Denoising + Contrast Enhancement |
| '7' | Denoising + Contrast Enhancement + Brightness Balance |

To evaluate the model, run
