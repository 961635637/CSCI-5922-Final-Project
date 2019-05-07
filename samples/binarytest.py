#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:57:42 2019

@author: group
"""


import numpy as np
import cv2
import math




def show(image, save = 'off', name = 'show_temp'):
    """
    显示并存储图片。如果save为on，则存储图片，否则仅显示。
    name为保存的图片名
    
    plt.figure()
    plt.title('The ' + name + ' image')
    plt.imshow(image)
    plt.show()
    """
    if save == 'on' or 'ON' or 'On' or 'oN':
        name = name + '.jpg'
        cv2.imwrite(name, image)

im_crop_32 = cv2.imread('/home/group/mask_RCNN/datasets/corn/val/crop_327.jpg')
    


hsv_green=cv2.cvtColor(im_crop_32,cv2.COLOR_BGR2HSV)

H, S, V = cv2.split(hsv_green)
LowerGreen = np.array([45, 70, 20])
UpperGreen = np.array([75, 255, 255])
mask = cv2.inRange(hsv_green, LowerGreen, UpperGreen)
GreenThings = cv2.bitwise_and(hsv_green, hsv_green, mask=mask)

show(mask, 'oN', 'mask')
show(GreenThings, 'ON', 'Green_crop32')