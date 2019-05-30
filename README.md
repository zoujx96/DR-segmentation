# DR-segmentation
Code Repo for Lesion Segmentation for Diabetic Retinopathy with Adversarial Learning

## Requirements:
PyTorch 1.0, TorchVision 0.2, Numpy, Scipy, PIL, skimage.

Please download [IDRiD dataset](https://idrid.grand-challenge.org/Data/) and put it under ```HEDNet_cGAN/data/```.

To prepare for preprocessing, please run ```HEDNet_cGAN/blackmask.py``` to get the mask for each fundus image. 
