To train a HEDNet model with conditional GAN to segment Hard Exudates using random seed 765, run ```python train_gan_ex.py --gan True --seed 765```.  
To train a HEDNet model with conditional GAN to segment Soft Exudates using random seed 765, run ```python train_gan_se.py --gan True --seed 765```.  
To train a HEDNet model with conditional GAN to segment Hemorrhages using random seed 765, run ```python train_gan_he.py --gan True --seed 765```.  
To train a HEDNet model with conditional GAN to segment Microaneurysms using random seed 765, run ```python train_gan_ma.py --gan True --seed 765```.  
When training HEDNet with cGAN, we apply all 3 preprocessing methods (Denoising + Contrast Enhancement + Brightness Balance).  
  
To evaluate the model on the test set, run ```python evaluate_model.py --seed 765 --lesion 'MA' --model results/models_ma/model.pth.tar``` for evaluating a saved model checkpoint on MA under ```results/``` using random seed 765. `results/models_ma/model.pth.tar` is the directory of the saved model checkpoint.
