#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Author  : Qiqi Xiao
# @Email     : xiaoqiqi177@gmail.com
# @File    : preprocess.py
# **************************************
#Inspired from: https://github.com/MasazI/clahe_python_opencv/blob/master/core.py

import sys
import os
import os.path
import cv2
import numpy as np


def clahe_gridsize(image_path, mask_path, denoise=False, brightnessbalance=None, cliplimit=None, gridsize=8):
    """This function applies CLAHE to normal RGB images and outputs them.
    The image is first converted to LAB format and then CLAHE is applied only to the L channel.
    Inputs:
      image_path: Absolute path to the image file.
      mask_path: Absolute path to the mask file.
      denoise: Toggle to denoise the image or not. Denoising is done after applying CLAHE.
      cliplimit: The pixel (high contrast) limit applied to CLAHE processing. Read more here: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
      gridsize: Grid/block size the image is divided into for histogram equalization.
    Returns:
      bgr: The CLAHE applied image.
    """
    bgr = cv2.imread(image_path)
    
    # brightness balance.
    if brightnessbalance:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask_img = cv2.imread(mask_path, 0)
        brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum()/255.)
        bgr = np.uint8(np.minimum(bgr * brightnessbalance / brightness, 255))

    # illumination correction and contrast enhancement.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
        bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    return bgr
