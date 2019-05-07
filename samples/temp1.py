# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:41:33 2018

@author: Yan
"""
import open_functions as ofs
import numpy as np
import cv2
import math
'''
im = ofs.white_edge('D:\\script\\BGREN\\rgb.jpg')
save_crop_dir = ofs.crop(im, color = 'rgb')
save_crop_dir = ofs.crop(im, color = 'gray')
ofs.convt_crop_seg(save_crop_dir, threshhold = 85)

'''
t = 99
while t <= 99:
    
    im_crop_32 = cv2.imread('C:\\Users\\Shane\\.spyder-py3\\crop_rgb\\crop_' + str(t) + '.jpg')
    
    
    #angle = np.arctan((68-70)/2960-2949)
    rotate_crop_32 = ofs.rotate(im_crop_32, -1.3, display ='on', save = 'on')
    hsv_green=cv2.cvtColor(rotate_crop_32,cv2.COLOR_BGR2HSV)
    
    H, S, V = cv2.split(hsv_green)
    LowerGreen = np.array([45, 70, 20])
    UpperGreen = np.array([75, 255, 255])
    mask = cv2.inRange(hsv_green, LowerGreen, UpperGreen)
    GreenThings = cv2.bitwise_and(hsv_green, hsv_green, mask=mask)
    
    
    ofs.show(im_crop_32)
    ofs.show(rotate_crop_32, 'On', 'rotate_crop_32')
    ofs.show(hsv_green, 'on', 'hsv_crop32')
    ofs.show(mask, 'oN', 'mask')
    ofs.show(GreenThings, 'ON', 'Green_crop32')
    '''

    #asd = cv2.line(GreenThings, (18,135), (2960,68), (0,255,0),3) #图像，起点，终点，颜色，线宽
    #asd = cv2.line(GreenThings, (2949,70), (2960,68), (0,0,255),3)


    #asd = cv2.line(GreenThings, (2960,32), (0,32), (0,0,255),3)
    #ofs.show(asd, 'ON', 'adasdasds')

    '''

    lines, temp = ofs.row_detection(mask, GreenThings)
    temp = ofs.pre_process(mask, lines, temp)

    #print(lines)
    #print(temp)

    width, temp = ofs.getWidth(mask, temp)
    '''
    i = 0
    position = []
    while i < len(temp):
        position.clear()
        k = 0
        position = ofs.findarea(mask, temp[i], temp[i + 1] + 1)
        while k < (len(position) - 1):
            cv2.line(rotate_crop_32, (position[k], temp[i]), (position[k+1], temp[i]),(0, 0, 255), 1)
            ofs.show(rotate_crop_32, 'on', 'mask1')
            cv2.line(rotate_crop_32, (position[k], temp[i+1]), (position[k+1], temp[i+1]),(0, 0, 255), 1)
            ofs.show(rotate_crop_32, 'on', 'mask1')
            k = k + 2
        i = i + 2
    print(width) 
    
    '''
    #width, temp = ofs.getWidth(mask, temp)
    #print(width)

    count = 0
    line = 0
    position = []
    while line < (len(temp) - 2 ):
        position.clear()
        k = 0
        if temp[line] < mask.shape[0] and temp[line+1] < mask.shape[0]:
            position = ofs.findarea(mask, temp[line], temp[line + 1] + 1)
            while k < (len(position) - 1):
                i = position[k]
                while i < position[k+1] - width:  #colend - width
                    white = 0
                    for m in range(i, i + width - 1):
                        for n in range(temp[line], temp[line + 1]):
                            if mask[n, m] == 255:
                                white = white + 1
                    total = width*(temp[line + 1] - temp[line] + 1)
                    percent = white/total
                    if percent < 0.05:
                        cv2.rectangle(rotate_crop_32, (i, temp[line]), (i + width - 1, temp[line+1]), (0, 0, 255), 1)   
                        ofs.show(rotate_crop_32, 'on', 'crop_' + str(t))
                        count = count + 1
                    i = i + width
                k = k + 2       
            line = line + 2
    t = t + 1
            #print(line)
            #print(count)
            #print(width)



'''
#计算角度

'''
    



            