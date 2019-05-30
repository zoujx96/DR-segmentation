# DR-segmentation
Code Repo for Lesion Segmentation for Diabetic Retinopathy with Adversarial Learning

## Requirements:
pytorch 1.0, torchvision 0.2, numpy 1.16, scikit-learn 0.20, cv2 3.4, PIL 5.1 and ipdb 0.12.
## Steps:
Please download [IDRiD dataset](https://idrid.grand-challenge.org/Data/) and put it under ```HEDNet_cGAN/data/```.

To prepare for preprocessing, please run ```HEDNet_cGAN/blackmask.py``` to get the mask for each fundus image. 
