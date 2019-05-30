# DR-segmentation
Open-sourced code for the paper *Improving Lesion Segmentation for Diabetic Retinopathy with Adversarial Learning*.

## Requirements:
pytorch 1.0, torchvision 0.2, numpy 1.16, scikit-learn 0.20, cv2 3.4, PIL 5.1 and ipdb 0.12.
## Steps:
1. Download [IDRiD dataset](https://idrid.grand-challenge.org/Data/) and put it under ```HEDNet_cGAN/data/```.

2. To prepare for preprocessing, please run ```HEDNet_cGAN/blackmask.py``` to get the mask for each fundus image. 

3. For training and evaluating **UNet** model, please go to ```UNet/```. For training and evaluating **HEDNet** model, please go to ```HEDNet/```. For training and evaluating **HEDNet with conditional GAN** model, please go to ```HEDNet_cGAN/```.
