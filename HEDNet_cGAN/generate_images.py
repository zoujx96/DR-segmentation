"""
File: train.py
Created by: Qiqi Xiao
Email: xiaoqiqi177<at>gmail<dot>com
"""

import sys
from torch.autograd import Variable
import os
from optparse import OptionParser
import numpy as np
import random
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, auc
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from PIL import Image

import config_gan as config
from unet import UNet
from hednet import HNNNet
from dnet import DNet
from utils import get_images
from dataset import IDRIDDataset
from torchvision import datasets, models, transforms
from transform.transforms_group import *
from torch.utils.data import DataLoader, Dataset
import argparse
np.set_printoptions(threshold=np.inf)
plt.switch_backend('Agg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_name = config.NET_NAME
image_size = config.IMAGE_SIZE
image_dir = config.IMAGE_DIR

def generate(model, eval_loader):
    model.eval()
    masks_soft = []
    masks_hard = []

    with torch.set_grad_enabled(False):
        for inputs, true_masks in eval_loader:
            inputs = inputs.to(device=device, dtype=torch.float)
            true_masks = true_masks.to(device=device, dtype=torch.float)
            bs, _, h, w = inputs.shape
            # not ignore the last few patches
            h_size = (h - 1) // image_size + 1
            w_size = (w - 1) // image_size + 1
            masks_pred = torch.zeros(true_masks.shape).to(dtype=torch.float)
            '''
            for i in range(h_size):
                for j in range(w_size):
                    h_max = min(h, (i + 1) * image_size)
                    w_max = min(w, (j + 1) * image_size)
                    inputs_part = inputs[:,:, i*image_size:h_max, j*image_size:w_max]
                    if net_name == 'unet':
                        masks_pred_single = model(inputs_part)
                    elif net_name == 'hednet':
                        masks_pred_single = model(inputs_part)[-1]
                    
                    masks_pred[:, :, i*image_size:h_max, j*image_size:w_max] = masks_pred_single

            masks_pred_softmax = F.softmax(masks_pred, dim=1).cpu().numpy().squeeze()
            masks_pred_final = masks_pred_softmax[1:,:,:].transpose(1, 2, 0)
            masks_pred_hard = (masks_pred_final > 0.5).astype(np.int)
            padding = np.zeros((masks_pred_hard.shape[0], masks_pred_hard.shape[1], 2)).astype(np.int)
            pred_image = (np.concatenate((masks_pred_hard, padding), axis=2) * 255).astype(np.uint8)
            pred_Image = Image.fromarray(pred_image)
            pred_Image.save('example_pred_GAN.jpg')
            '''

            gts = true_masks.cpu().numpy().squeeze()
            gts = gts[1:,:,:].transpose(1, 2, 0).astype(np.int)
            padding = np.zeros((gts.shape[0], gts.shape[1], 2)).astype(np.int)
            pred_image = (np.concatenate((gts, padding), axis=2) * 255).astype(np.uint8)
            pred_Image = Image.fromarray(pred_image)
            pred_Image.save('IDRiD_55_HE.jpg')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model', type=str)
    parser.add_argument('--lesion', type=str)
    args = parser.parse_args()
    #Set random seed for Pytorch and Numpy for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if net_name == 'unet':
        model = UNet(n_channels=3, n_classes=2)
    else:
        model = HNNNet(pretrained=True, class_number=2)

    resume = args.model

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch'] + 1
        start_step = checkpoint['step']
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model.load_state_dict(checkpoint['g_state_dict'])
        print('Model loaded from {}'.format(resume))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    model.to(device)

    test_image_paths, test_mask_paths = get_images(image_dir, config.PREPROCESS, phase='test')

    if net_name == 'unet':
        test_dataset = IDRIDDataset(test_image_paths, test_mask_paths, config.LESION_IDS[args.lesion])
    elif net_name == 'hednet':
        test_dataset = IDRIDDataset(test_image_paths, test_mask_paths, config.LESION_IDS[args.lesion], \
            transform=Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))

    test_loader = DataLoader(test_dataset, 1, shuffle=False)
    generate(model, test_loader)

    