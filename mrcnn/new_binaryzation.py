#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:40:24 2019

@author: yan
"""

import cv2
import numpy as np

img = cv2.imread('C:\\crop.jpg')


def new_bina(img):
    mask = np.zeros((img.shape[0], img.shape[1]))
    green_image = img[ : , : , 1]
    red_image = img[ : , : , 2]
    blue_image = img[ : , : , 0]
    Rmax = 0
    Gmax = 0
    Bmax = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if green_image[i][j] > Gmax:
                Gmax = green_image[i][j]
            if red_image[i][j] > Rmax:
                Rmax = red_image[i][j]
            if blue_image[i][j] > Bmax:
                Bmax = blue_image[i][j]
    
    G1 = green_image / Gmax
    R1 = red_image / Rmax
    B1 = blue_image / Bmax
    
    sum = R1 + B1 + G1
    
    r = R1 / sum
    g = G1 / sum
    b = B1 / sum
    
    exg = 2*g - r - b
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if exg[i][j] > 0.1:
                mask[i][j] = 255
            else:
                mask[i][j] = 0
    return mask
mask = new_bina(img)       
cv2.imwrite('C:\\crop_revised.jpg', mask)




'''
mask = np.zeros((img.shape[0], img.shape[1]))
green_image = img[ : , : , 1]
red_image = img[ : , : , 2]
blue_image = img[ : , : , 0]

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if green_image[i][j] > red_image[i][j] and green_image[i][j] > blue_image[i][j]:
            mask[i][j] = 255
        else:
            mask[i][j] = 0

'''
