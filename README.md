# DR-segmentation
Open-sourced code for the paper 
> Qiqi Xiao, Jiaxu Zou, Muqiao Yang, Alex Gaudio, Kris Kitani, Asim Smailagic, Pedro Costa and Min Xu, Improving Lesion Segmentation for Diabetic Retinopathy using Adversarial Learning, ICIAR, 2019

Please see our [Presentation at Conference](https://docs.google.com/presentation/d/1T4w1mRxClnDm0sGmlbDa8FRYbV6hLG-nA12GLtL9iwo/edit#slide=id.f18e43d3-c5f6-11e9-bf5f-cb4e139218f9)
## Requirements:
pytorch 1.0, torchvision 0.2, numpy 1.16, scikit-learn 0.20, cv2 3.4, PIL 5.1 and ipdb 0.12.
## Steps:
1. Download [IDRiD dataset](https://idrid.grand-challenge.org/Data/) and put it under ```HEDNet_cGAN/data/```.

2. To prepare for preprocessing, please run ```HEDNet_cGAN/blackmask.py``` to get the mask for each fundus image. 

3. For training and evaluating **UNet** model, please go to ```UNet/```. For training and evaluating **HEDNet** model, please go to ```HEDNet/```. For training and evaluating **HEDNet with conditional GAN** model, please go to ```HEDNet_cGAN/```. 

