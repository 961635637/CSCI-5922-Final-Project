#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:37:28 2019

@author: group
"""

import open_functions as ofs
import numpy as np
import cv2
import math

im = ofs.white_edge('/home/group/Downloads/RGB.jpg')
save_crop_dir = ofs.crop(im, n = 20, color = 'rgb')
#save_crop_dir = ofs.crop(im, color = 'gray')
ofs.convt_crop_seg(save_crop_dir, threshhold = 85)