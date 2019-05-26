"""
File: blackmask.py
Created by: Qiqi Xiao
Email: xiaoqiqi177<at>gmail<dot>com
Description: Get and save the mask for each original image.
"""

import os
import glob
import cv2
import numpy as np
import config

image_dir = config.IMAGE_DIR
for setname in ['TrainingSet', 'TestingSet']:
    mask_dir = os.path.join(image_dir, 'Groundtruths', setname, 'Mask')
    os.mkdir(mask_dir)

    imgs_ori = glob.glob(os.path.join(image_dir, 'OriginalImages/'+setname+'/*.jpg'))

    for image_path in imgs_ori:
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        black_mask = np.uint8((image_gray > 15)*255.)
        ret, thresh = cv2.threshold(black_mask, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.ones(image.shape[:2], dtype='uint8')*255
        cn = []
        for contour in contours:
            if len(contour) > len(cn):
                cn = contour
        cv2.drawContours(mask, [cn], -1, 0, -1)
        image_name = os.path.split(image_path)[-1].split('.')[0]
        mask_path = os.path.join(mask_dir, image_name+'_MASK.tif')
        cv2.imwrite(mask_path, mask)
