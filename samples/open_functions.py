# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:51:52 2018

@author: Yan
"""
from libtiff import TIFF
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil

def read_one_channel(path):
    im = tiff.imread(path) + 0.0 # to convert the unsigned int16 to float
    channel_name =path.split("_")[-1].split(".")[0]
    
    
    print('The size of the original image with only ' + channel_name + ' channel is: ' + str(im.shape) + '.')
    #print('The number of nonzero valus is ' + str(np.nonzero(im)[0].shape) + '.')
    temp = np.amax(im[:,:]) / 255
    im[:,:] = im[:,:] / temp
    im = im.astype(int)
    plt.figure()
    plt.title('The original image with only ' + channel_name + ' channel')
    plt.imshow(im)
    plt.show()
    return im, channel_name



def show_tiny_one_channel(im, channel_name):
    """
    显示单通道的一个很小的切片图像 500*1000
    """
    print('Let us zoom in and see one tiny piece on the image with only ' + channel_name + ' channel')
    plt.figure()
    plt.title('The tiny piece on the image with only ' + channel_name + ' channel')
    row = int(im.shape[0]/2)
    col = int(im.shape[1]/2)
    plt.imshow(im[row:row + 500, col:col + 1000])
    plt.show()
    im_tiny_one_channel = im[row:row + 500, col:col + 1000]
    return im_tiny_one_channel
    
def show_tiny_bgr_channel(im_new_bgr):
    """
    显示RGB的一个很小的切片图像  500*1000
    """
    print('Let us zoom in and see one tiny piece on the RGB image')
    plt.figure()
    plt.title('The tiny piece on the RGB image')
    row = int(im_new_bgr.shape[0]/2)
    col = int(im_new_bgr.shape[1]/2)
    plt.imshow(im_new_bgr[row:row + 500, col:col + 1000, :])
    plt.show()
    im_tiny = im_new_bgr[row:row + 500, col:col + 1000, :]
    return im_tiny

	
	
	
def delete_useless(im, channel_name):
    """
    切除四周多余的边，注意这个只能用于单一通道
    """
    # Crop the useless part of the image
    for row_up in range(im.shape[0]):
        if np.sum(im[row_up,:]) > 0:
            break
    
    for row_down in range(im.shape[0]-1,-1,-1):
        if np.sum(im[row_down,:]) > 0:
            break
    
    for col_up in range(im.shape[1]):
        if np.sum(im[:,col_up]) > 0:
            break
        
    for col_down in range(im.shape[1]-1,-1,-1):
        if np.sum(im[:,col_down]) > 0:
            break
    
    if row_up > 0:
        assert(np.sum(im[row_up, :])) > 0
        assert(np.sum(im[row_up - 1, :])) == 0
    
    if row_down < im.shape[0]:
        assert(np.sum(im[row_down, :])) > 0
        assert(np.sum(im[row_down + 1, :])) == 0
        
    if col_up > 0:
        assert(np.sum(im[:, col_up])) > 0
        assert(np.sum(im[:, col_up -1])) == 0
        
    if col_down < im.shape[1]:
        assert(np.sum(im[:, col_down])) > 0
        assert(np.sum(im[:, col_down + 1])) == 0
    
    im_new = im[row_up:row_down + 1, col_up:col_down + 1]
    print('The size of the new image with the ' + channel_name + ' channel is: ' + str(im_new.shape) + '.')
    
    plt.figure()
    plt.title('The new image with only the ' + channel_name + ' channel deleted the useless part')
    plt.imshow(im_new)
    plt.show()
    
    im_tiny_one_channel = show_tiny_one_channel(im_new, channel_name)
    saved_name = 'gray_' + channel_name + '.jpg'
    cv2.imwrite(saved_name, im_tiny_one_channel)
    
    return im_new

def white_edge(original_location):
    """
    把图片中边缘处黑色的区域转换成白色的区域RGB
    """
    im = cv2.imread(original_location)
    plt.figure()
    plt.title('The original image:')
    plt.imshow(im)
    plt.show()
    
    im[im == 0] = 255
    plt.figure()
    plt.title('The new image:')
    plt.imshow(im)
    plt.show()
    cv2.imwrite('rgb_new.jpg', im)
    return im

def mix_BGR(im_BGR):
    """
    将三个单通道合成一个RGB图像，注意内存溢出问题
    """
    temp = np.amax(im_BGR[:,:,0]) / 255
    im_BGR[:,:,0] = im_BGR[:,:,0] / temp
    temp = np.amax(im_BGR[:,:,1]) / 255
    im_BGR[:,:,1] = im_BGR[:,:,1] / temp
    temp = np.amax(im_BGR[:,:,2]) / 255
    im_BGR[:,:,2] = im_BGR[:,:,2] / temp
    im_BGR = im_BGR.astype(int)
    plt.figure()
    plt.title('The RGB image')
    plt.imshow(im_BGR)
    plt.show()
    show_tiny_bgr_channel(im_BGR)
    return im_BGR


def segment_gray(ori_location, threshhold = 85, name = 'segment.jpg'):
    """
    在灰度图像中只保留绿苗，其余的删除
    threshhold 默认85
    location为原灰度图的位置
    name 处理后图像的名字
    """
    im = cv2.imread(ori_location, cv2.IMREAD_GRAYSCALE)
    im[im > threshhold] = 255
    seg_dir = 'seg_gray/'
    seg_location = seg_dir + name
    cv2.imwrite(seg_location, im)
    return im
    
def crop(im, n = 10, color = 'rgb'):
    """
    把一张图切割成n*n张子图。
    原图片默认为彩色，如果为灰度，请输入color='gray'
    默认切割成10×10张子图。如果需要切割成其他数量，eg:请输入n=8(切割为64张)
    如果segment = ON，那么只保留绿色，其余去除。默认为OFF
    """
    if color == 'rgb':
        save_crop_dir = '/home/group/mask_RCNN/samples/balloon/crop_rgb/'
    elif color == 'gray':
        save_crop_dir = '/home/group/mask_RCNN/samples/balloon/crop_gray/'
 
    if os.path.isdir(save_crop_dir):
        shutil.rmtree(save_crop_dir)

    if not os.path.exists(save_crop_dir):
        os.mkdir(save_crop_dir)

    #im = cv2.imread(img_location)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    print('The dimention of original image is ' + str(im_gray.shape) + '.')
    print('There will be ' + str(n*n) + ' cropped images saved to ' + save_crop_dir + '.')
    
    #n = 10
    #part_dimension is to store the dimension of cropped images
    #part_dimension[0] is the dimension of the first n-1 cropped image, and part_dimension[1] is the last cropped image
    part_dimension = np.zeros((2, 2)).astype(int)
    part_dimension[:,0] = part_dimension[:,0] + int(im_gray.shape[0] / n)
    part_dimension[1,0] = part_dimension[1,0] + im_gray.shape[0] % n
    part_dimension[:,1] = part_dimension[:,1] + int(im_gray.shape[1] / n)
    part_dimension[1,1] = part_dimension[1,1] + im_gray.shape[1] % n
    print('The total number of cropped images are ' + str(n*n) + '.')
    print('The dimension of the first ' + str(n-1) + ' cropped images are ' + str(part_dimension[0]) + ';')
    print('The dimension of the last cropped image is ' + str(part_dimension[1]) + '.')
    
    #print(part_dimension)

    if color == 'gray':
        im_new = im_gray
    else:
        im_new = im
    i = 0
    row = 0
    count = 0
    for i in range(0,n):

        #print ("Row No. is " + str(row))
        col = 0
        for j in range (0,n):
            #print (col)
            cropped_name = 'crop_' + str(count) +'.jpg'
            cropped_location = save_crop_dir + cropped_name
            
            if i < n - 1:
                cv2.imwrite(cropped_location, im_new[row:row+part_dimension[0,0], col:col+part_dimension[0,1]])
                col = col + part_dimension[0,1]
            elif i == n - 1:
                cv2.imwrite(cropped_location, im_new[row:row+part_dimension[1,0], col:col+part_dimension[1,1]])
                col = col + part_dimension[1,1]

            
            count = count + 1
        
        #print ("i is " +str(i))
        i = i + 1    
        row = row + part_dimension[0,0] 
    return save_crop_dir

def convt_crop_seg(save_crop_dir, threshhold = 85):
    """
    将切割好的全部灰度图片中只保留青苗
    """
    seg_dir = 'seg_gray/'
    if os.path.isdir(seg_dir):
        shutil.rmtree(seg_dir)
    
    if not os.path.exists(seg_dir):
        os.mkdir(seg_dir)
        
    for file in os.listdir(save_crop_dir):
        if os.path.isfile(os.path.join(save_crop_dir,file))==True:
            ori_location = os.path.join('crop_gray/', file)
            name_location = 'seg_' + file
            segment_gray(ori_location, threshhold, name = name_location)

def show(image, save = 'off', name = 'show_temp'):
    """
    显示并存储图片。如果save为on，则存储图片，否则仅显示。
    name为保存的图片名
    """
    plt.figure()
    plt.title('The ' + name + ' image')
    plt.imshow(image)
    plt.show()
    if save == 'on' or 'ON' or 'On' or 'oN':
        name = name + '.jpg'
        cv2.imwrite(name, image)

def rotate(image, angle, display = 'off', save = 'off', name = 'rotate_temp'):
    """
    将照片旋转多少度（angle）
    image是原照片，display是否显示，save是否保存，name旋转后的照片名字
    """
    rows,cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    res = cv2.warpAffine(image,M,(cols,rows))
    if display == 'on' or 'ON' or 'On' or 'oN':
        show(res, 'on')
    if save == 'on' or 'ON' or 'On' or 'oN':        
        name = name + '.jpg'
        cv2.imwrite(name, res)
    return res

def delete_closed_line(temp):
    """
    删除相邻颜色相同的线（红或蓝）所在行
    name为up或down
    """
    temp1 = []
    for i in range(1, len(temp)-2):
        if temp[i] - temp[i-1] <3:
            temp1.append(temp[i])
            i = i +2
        if temp[i] - temp[i-1] >8:
           temp1.append(temp[i-1])
           i = i + 1
           '''
           if temp[i+1] - temp[i] >1:
               temp1.append(temp[i])
               i = i + 1
           '''
    '''           
    if temp[-2] - temp1[-1] <3:
        temp1.remove(temp1[-1])
        temp1.append(temp[-2])
    elif temp[-2] - temp1[-1] >8:
        temp1.append(temp[-2])
    
    if temp[-1] - temp1[-1] <3:
        temp1.remove(temp1[-1])
        temp1.append(temp[-1])
    elif temp[-1] - temp1[-1] >8:
        temp1.append(temp[-1])
    '''       
    new = []
    for i in temp1:
        if not i in new:
            new.append(i)
    print('You will delete the closed lines with the same color.') 
    new.sort()       
    return new

def findarea(mask, rowstart, rowend):
    '''
    若图像有白色区域，寻找非白色区域
    返回此区域开始结束点
    '''
    #找开始区域
    position = []
    position.clear()
    if rowstart < mask.shape[0] and rowend < mask.shape[0]:
        i = 0
        label = False
        while i < mask.shape[1]:
            p = 0
            for p in range(rowstart, rowend):
                if mask[p, i] == 255:
                    label = True
            if label == True:
                break
            else:
                i = i + 1
        position.append(i)
        #print(i)
        #找结束区域
        j = mask.shape[1] - 1
        label = False
        while j > i:
            p = 0
            for p in range(rowstart, rowend):
                if mask[p, j] == 255:
                    label = True
            if label == True:
                break
            else:
                j = j - 1
        position.append(j)
        
        while j > i + 50:
            countw = 0 
            label = False
            for p in range(rowstart, rowend):
                for q in range(i, i + 50):
                    if mask[p, q] == 255:
                        countw = countw + 1
            #whitepercent = countw/(50*(rowend - rowstart))
            if countw <= 60 and j > i + 150:
                countw = 0 
                #if rowstart > 50 and rowend > mask.shape[0] - 50:
                for p in range(rowstart, rowend):
                    #print(i)
                    for q in range(i + 50, i + 150):
                        if mask[p, q] == 255:
                            countw = countw + 1                 
                if countw <= 2:
                    position.append(i)
                    label = True
                '''
                elif rowstart < 50 and rowend < mask.shape[0] - 50:
                    for p in range(rowstart, rowend + 50):
                        #print(i)
                        for q in range(i + 50, i + 150):
                            if mask[p, q] == 255:
                                countw = countw + 1                 
                    if countw <= 15:
                        position.append(i)
                        label = True
                else:
                    for p in range(rowstart - 20, rowend + 20):
                        #print(i)
                        for q in range(i + 50, i + 150):
                            if mask[p, q] == 255:
                                countw = countw + 1                 
                    if countw <= 50:
                        position.append(i)
                        label = True
                '''
            if label == True:
                break
            i = i + 50
            
        while j > i + 50:
            countw = 0 
            label = False
            for p in range(rowstart, rowend):
                for q in range(j - 50, j):
                    if mask[p, q] == 255:
                        countw = countw + 1
            if countw <= 60 and j > i + 150:
                countw = 0 
                #if rowstart > 50 and rowend > mask.shape[0] - 50:
                for p in range(rowstart, rowend):
                    for q in range(j - 150, j - 50):
                        if mask[p, q] == 255:
                            countw = countw + 1
                if countw <= 2:
                    position.append(j)
                    label = True
                '''
                elif rowstart < 50 and rowend < mask.shape[0] - 50:
                    for p in range(rowstart, rowend + 50):
                        for q in range(j - 150, j - 50):
                            if mask[p, q] == 255:
                                countw = countw + 1
                    if countw <= 15:
                        position.append(j)
                        label = True
                else:
                    for p in range(rowstart - 20, rowend + 20):
                        for q in range(j - 150, j - 50):
                            if mask[p, q] == 255:
                                countw = countw + 1
                    if countw <= 50:
                        position.append(j)
                        label = True
                '''
            if label == True:
                break
            j = j - 50
                
    position.sort()
    print('area number = ', position)
    return position
    '''
    while i < mask.shape[1] - 80 :
            label = False
            for p in range(rowstart, rowend):
                for q in range(i, i + 80):
                    if mask[p, q] == 255:
                        label = True
                        break
    
            if label == False:
                break
            i = i + 80
        end = i
    '''
    


def row_detection(mask, GreenThings):
    """
    形参为一张图片，返回一个字典，包含配对好的上下确界
    画苗的上确界和下确界；
    红色为上确界，蓝色为下确界
    """
    Threshold_Out = 255*100 #如果该行的和小于这个值，那么意味着此处是空白区域
    Threshold_In = 255*mask.shape[1]/100 #如果该行的和大于这个值，那么意味着此处是青苗区域
    #在Up和Down之间的行全部都是苗（包括该行）
    up_temp = [] #存储上确界的行（红线）
    down_temp = [] #存储下确界的行（蓝线）
    for row in range(1, mask.shape[0]-1):
        if np.sum(mask[row,:]) <= Threshold_Out and np.sum(mask[row+1,:]) >= Threshold_In:
            asd = cv2.line(GreenThings, (mask.shape[1],row), (0,row), (0,0,255),2) #画上确界红线
            up_temp.append(row)
        if np.sum(mask[row-1,:]) >= Threshold_In and np.sum(mask[row,:]) <= Threshold_Out:
            asd = cv2.line(GreenThings, (mask.shape[1],row), (0,row), (255,0,0),2) #画下确界蓝线
            down_temp.append(row)
    
    
    up = delete_closed_line(up_temp)
    down1 = delete_closed_line(down_temp)
    down = delete_closed_line(down1)
    del up_temp, down_temp, down1
    #print(up)
    #print(down)
    
    temp = up + down
    temp.sort()
    
    
    for i in temp:
        if i >= up[0]:
            break
        else:
            temp.remove(i)
    for i in temp[::-1]:
        if i <= down[-1]:
            break
        else:
            temp.remove(i)
    dict = {}
    for i in temp:
        if i in up:
            dict[i] ='red'
        else:
            dict[i] ='blue'
    print(dict)

    #temp列表查重
    m = 0
    while m == 0:
        k = 0
        for i in range(len(temp)):
            if temp[i - 1] == temp[i]:
                temp.pop(i)
                k = 1
                break
        if k == 1:
            continue
        else:
            break       
    #########
    #删除多余的没用的行
    
    #dict.pop(temp[1])
    #判断开头蓝色 
    for p in range(len(temp)):
        if dict[temp[p]] != 'red':
            dict.pop(temp[p])        
        else:
            break
    
   
    #判断结尾红色
    temp.reverse()
    for p in range(len(temp)):
        if dict[temp[p]] != 'blue':
            dict.pop(temp[p])        
        else:
            break
    temp.reverse()
     
    temp = list(dict.keys())
    #判断红色重复
    for p in range(len(temp)):
        if dict[temp[p]] == 'red':
            if dict[temp[p-1]] == 'red':
                dict.pop(temp[p-1])

    temp = list(dict.keys())
    print(dict)
    #判断蓝色重复
    for p in range(len(temp)-1):
        if dict[temp[p]] == 'blue':
            if dict[temp[p+1]] == 'blue':
                dict.pop(temp[p])
    
    temp = list(dict.keys())
    return dict, temp


def pre_process(mask, lines, temp):      
    '''
    预处理图片，清楚的分出每一排， mask 二值化， lines temp 红蓝线数组元祖
    返回列表包含所有红蓝线
    '''
    #确认排与排间距
    
    i = 1
    space = []
    while i < (len(temp) - 1):
        space.append(temp[i+1] - temp[i])
        i = i + 2
    #print(space)
    space.sort()
    if len(space) % 2 == 0:      
        spacewidth = space[int((len(space)/2))]
    else:
        spacewidth = space[int((len(space)-1)/2)]
    print(space)
    print(temp)
    print(spacewidth)
    '''
    i = 1
    while i < (len(temp) - 1):
        if (temp[i+1] - temp[i]) < (spacewidth - 12):
            lines.pop(temp[i+1])
            lines.pop(temp[i])
            temp.pop(i+1)
            temp.pop(i)
        i = i + 2
    '''
    #print(temp)
    #width = ofs.getWidth(mask, temp)
    #print(width)
    print(temp)
    #确认每一行苗宽度然后调整框高度
    height = []
    height.clear()
    i = 0
    while i < len(temp):
        if temp[i+1] - temp[i] > 10 and temp[i+1] - temp[i] < 20:       
            height.append(temp[i+1] - temp[i])
        i = i + 2
    height.sort()
    if len(height) == 0:
        height.append(13)
    print(height)
    if len(height) % 2 == 0:      
        strdheight = height[int((len(height)/2))] + 10
    else:
        strdheight = height[int((len(height)-1)/2)] + 10
    i = 0
    while i < len(temp):
        if (temp[i+1] - temp[i] <= strdheight - 3) or (temp[i+1] - temp[i] >= strdheight + 3):
            temp[i + 1] = temp[i] + strdheight
        #print(i)
        i = i + 2
    
    
    '''
    if temp[0] < 0:
        if temp[1] >= strdheight - 5:
            temp[0] = 0
        else:
            temp.pop(0)
            temp.pop(0)      
    
    height.clear()
    i = 0
    while i < len(temp):
        height.append(temp[i+1] - temp[i])
        i = i + 2
       
    i = 0
    pixelpost = []
    print(temp)
    while i < len(temp):
        pixelpost.clear()
        count = 0
        
        if i == 0:
            for temp[i] in range(0, temp[i+2] - height[i]):
                temp[i + 1] = temp[i] + height[i]
                count = 0
                for m in range(temp[i], (temp[i+1]+1)):
                    for n in range(mask.shape[1]):
                        if mask[m, n] == 255:
                            count = count + 1
                pixelpost.append(count)
            temp[i] = pixelpost.index(max(pixelpost))
            temp[i+1] = temp[i] + height[i]
        
        elif i == len(temp) - 2:
            for startrowp in range(temp[i-1] + 5, mask.shape[0] - height[int(i/2)]):              
                endrowp = startrowp + height[int(i/2)]
                count = 0
                #print('startrow = ', startrowp)
                #print('endrow = ', endrowp)
                for m in range(startrowp, (endrowp+1)):
                    for n in range(mask.shape[1]): 
                        if mask[m, n] == 255:        
                            count = count + 1
                pixelpost.append(count)
            temp[i] = pixelpost.index(max(pixelpost)) + temp[i-1] + 5
            temp[i+1] = temp[i] + height[int(i/2)]
        
        else :
            #print('temp[i] = ', temp[i])
            #print('temp[i-1] = ', temp[i-1])
            for startrowp in range(temp[i-1] + 5, temp[i+2] - height[int(i/2)]):              
                endrowp = startrowp + height[int(i/2)]
                count = 0
                #print('startrow = ', startrowp)
                #print('endrow = ', endrowp)
                for m in range(startrowp, (endrowp+1)):
                    for n in range(mask.shape[1]): 
                        if mask[m, n] == 255:        
                            count = count + 1
                pixelpost.append(count)
            temp[i] = pixelpost.index(max(pixelpost)) + temp[i-1] + 5
            temp[i+1] = temp[i] + height[int(i/2)]
            print(temp[i])
            print(temp[i+1])
        
        
        i = i + 2                 
    '''
    
    
    '''
    if strdheight < 10:
        if len(height) % 2 == 0:      
            strdheight = height[int((len(height)/2))]
        else:
            strdheight = height[int((len(height)-1)/2)]
        
        
        if height[(len(height)-1)] >= 10:         
            strdheight = height[(len(height)-1)]
        
        else:
            strdheight = 13
    
    #print(temp)
    i = 0
    while i < len(temp):
        if (temp[i+1] - temp[i]) != (strdheight-1):
            if(i == 22):
                temp[i+1] = temp[i] + strdheight - 1
            else:
                temp[i] = temp[i+1] - strdheight + 1
        i = i + 2
   '''
    #再次确认排与排间距
    i = 1
    space.clear()
    while i < (len(temp) - 1):
        space.append(temp[i+1] - temp[i])
        i = i + 2
    space.sort()
    if len(space) % 2 == 0:      
        spacewidth = space[int((len(space)/2))]
    else:
        spacewidth = space[int((len(space)-1)/2)]
     
    #查找有无遗漏
    label = False
    '''
    if spacewidth > 25 or spacewidth < 10:
        spacewidth = 24
    if strdheight > 15 or strdheight < 5:
        strdheight = 12
    '''
    
    while label == False:
        i = 1
        while i < (len(temp) - 1):
            if (temp[i+1] - temp[i]) > (1.1*spacewidth + strdheight):  #1.5
                temp.append(temp[i] + spacewidth)
                temp.append(temp[i + 1] - spacewidth)
                temp.sort()
                label = False
                break
            else:
                label = True 
            
            if (temp[i+1] - temp[i]) < (spacewidth - 2):
                temp[i] = temp[i + 1] - spacewidth
                temp[i - 1] = temp[i] - strdheight
                label = False
                break
            else:
                label = True
            
            i = i + 2
    
    #判断首排苗是否值得留下
    if temp[0] < 0:
        if temp[1] >= 9:
            temp[0] = 0
        else:
            temp.pop(0)
            temp.pop(0)
    #判断最后一排苗是否值得留下
    if temp[-1] >= mask.shape[0]:
        temp.pop(-1)
        temp.pop(-1)
    
    #最后的筛选
    i = 0
    #temp[0] = temp[0] - 7   #6 运行需要打开
    while i < (len(temp)-1):
        if i % 2 == 0:
            temp[i + 1] = temp[i] + strdheight
        else:
            if temp[i] + spacewidth < mask.shape[0]:
                temp[i + 1] = temp[i] + spacewidth
        i = i + 1
    
    height.clear()
    i = 0
    while i < len(temp):
        height.append(temp[i+1] - temp[i])
        i = i + 2
       
    i = 0
    pixelpost = []
    print(temp)
    while i < len(temp):
        pixelpost.clear()
        count = 0
        
        if i == 0:
            for temp[i] in range(0, temp[i+2] - height[i]):
                temp[i + 1] = temp[i] + height[i]
                count = 0
                for m in range(temp[i], (temp[i+1]+1)):
                    for n in range(mask.shape[1]):
                        if mask[m, n] == 255:
                            count = count + 1
                pixelpost.append(count)
            if len(pixelpost): 
                temp[i] = pixelpost.index(max(pixelpost))
                temp[i+1] = temp[i] + height[i]
        
        elif i == len(temp) - 2:
            for startrowp in range(temp[i-1] + 5, mask.shape[0] - height[int(i/2)]):              
                endrowp = startrowp + height[int(i/2)]
                count = 0
                #print('startrow = ', startrowp)
                #print('endrow = ', endrowp)
                for m in range(startrowp, (endrowp+1)):
                    for n in range(mask.shape[1]): 
                        if mask[m, n] == 255:        
                            count = count + 1
                pixelpost.append(count)
            if len(pixelpost):     
                temp[i] = pixelpost.index(max(pixelpost)) + temp[i-1] + 5
                temp[i+1] = temp[i] + height[int(i/2)]
            print(temp[i])
            print(temp[i+1])
        
        else :
            #print('temp[i] = ', temp[i])
            #print('temp[i-1] = ', temp[i-1])
            for startrowp in range(temp[i-1] + 3, temp[i-1] + spacewidth + strdheight + 3):   #           temp[i+2] - height[int(i/2)] - 3
                
                endrowp = startrowp + height[int(i/2)]
                if endrowp >= mask.shape[0]:
                    endrowp = mask.shape[0] - 1
                count = 0
                #print('startrow = ', startrowp)
                #print('endrow = ', endrowp)
                for m in range(startrowp, (endrowp+1)):
                    for n in range(mask.shape[1]): 
                        if mask[m, n] == 255:        
                            count = count + 1
                pixelpost.append(count)
            if len(pixelpost): 
                temp[i] = pixelpost.index(max(pixelpost)) + temp[i-1] + 5
                temp[i+1] = temp[i] + height[int(i/2)]
            print(temp[i])
            print(temp[i+1])
        
        
        i = i + 2  
    temp.sort()
    return temp
        

def getWidth(mask, temp):
    '''
    mask 二值化， temp 红蓝线列表
    返回苗的宽度
    '''
#求每行有多少苗
    n = 0
    count = 0
    label = False
    Blackpix = 0
    Whitepix = 0
    label2 = False
    widthls = []
    position = []
    
    #change = 1
    #for i in range(len(lines)):
    

    while n <= (len(temp) - 1):
        k = 0
        position.clear()
        position = findarea(mask, temp[n], (temp[n+1]+1))
        while k < (len(position) - 1):
            for q in range(position[k], position[k + 1]):
                label2 = False
                for p in range(temp[n], (temp[n+1]+1)):      
                    if mask[p, q] == 255:
                        #if (mask[p + 4, q] == 255) or (mask[p - 4, q] == 255) or (mask[p, q + 4] == 255) or (mask[p, q - 4] == 255):
                        if label == False:
                            label = True           
                        Whitepix = Whitepix + 1
                        label2 = True
                        
                    #elif (mask[p, q] == 255) and (label == True):
                        #Whitepix = Whitepix + 1
                           
                    elif mask[p, q] == 0:
                        if label == True:        
                            Blackpix = Blackpix + 1
                            if p == temp[n+1] and mask[p, q] == 0 and label2 == False:
                                label = False
                                percent = Whitepix/(Blackpix + Whitepix)
                                if percent > 0.15:
                                    count = count + 1
                                    #print(m)
                                    #print(q)
                                    Blackpix = 0
                                    Whitepix = 0
            k = k + 2
        if count > 0:   
            if len(position) == 2:
                widthls.append(int((position[1] - position[0] + 1) / count))
            elif len(position) == 4:
                widthls.append(int((position[3] - position[2] + position[1] - position[0] + 2) / count))
        count = 0
        n = n + 2
        '''
        if count < 10:
            n = n - 2
            temp[n] =  temp[n] - 2
            temp[n+1] =  temp[n+1] - 2
            temp[n+2] =  temp[n+2] - 3
            temp[n+3] =  temp[n+3] - 3
           # change = n + 2
        else:          
            widthls.append(int((colend - colstart + 1) / count))
            count = 0
        '''
        
        
    widthls.sort()
    if len(widthls) % 2 == 0:      
        width = widthls[int((len(widthls)/2))]
    else:
        width = widthls[int((len(widthls)-1)/2)]
    #print(widthls)
    '''   
    if width >= 30:
        temp = reprocess(temp)
    '''
    return width, temp
'''
def reprocess(temp):  
    i = 0
    while i < len(temp):
        temp[i] = temp[i] - 3
        i = i + 1
    return temp
'''
def merge(crop_path, N, name = 'Merged_Image.jpg'):
    """
    把切割的N*N的照片合成一张
    crop_path为存放切割照片的绝对路径
    N*N为切割的数量，name为保存的合成照片的名字
    """
    for dirName, subdirList, fileList in os.walk(crop_path):
        for fname in fileList:
            fname_extension = fname.split('.')[1]
            if fname_extension == '.jpg':
                fileList.remove(fname)
    image_count = len(fileList)
    print ('The number of images in this location is ' + str(image_count) + '.')
    
    if np.power(N, 2) != image_count:
        print('The number of images is not equal to N*N. Please double check your cropped images.')
    assert(image_count == np.power(N, 2))
    
    rows = []
    for i in range(N): # i is the row from 0 to N-1
        for j in range(0, N): # j is the column from 0 to N-1
            if j == 0:
                first_image = cv2.imread(crop_path + '\\crop_'+ str(i * 10 + j) + '.jpg')       
                #print(first_image.shape[1])   
            else:
                next_image = cv2.imread(crop_path + '\\crop_'+ str(i * 10 + j) + '.jpg')
                first_image = np.concatenate((first_image, next_image), axis=1)
                #print(next_image.shape[1])
            print('crop_'+ str(i * 10 + j) + '.jpg is done')
        print("The columns in No. " + str(i) + " row is complete merging.")
        rows.append(first_image)
    
    temp = []
    for row in rows:
        temp.append(row.shape[1])
    column = max(temp)
    
    new_rows = []
    
    for i in range(N):
        add_column = column - rows[i].shape[1]
    
        if i == 0:
            if add_column != 0:
                add_area = np.zeros((rows[i].shape[0], add_column, 3)).astype(int)
                new_rows.append(np.concatenate((rows[i], add_area), axis=1))
            else:
                new_rows.append(rows[i])
            first_row = new_rows[i]
        else:
            if add_column != 0:
                add_area = np.zeros((rows[i].shape[0], add_column, 3)).astype(int)
                new_rows.append(np.concatenate((rows[i], add_area), axis=1))
            else:
                new_rows.append(rows[i])
            
            next_row = new_rows[i]
            #print(count)
            first_row = np.concatenate((first_row, next_row), axis=0)
        print("The No. " + str(i) + " row is complete merging.")
    print("The shape of merged image is " + str(first_row.shape))
        
    print("Let us see the merged image!")
    plt.figure()
    plt.title('The Merged image')
    plt.imshow(first_row)
    plt.show()
    
    cv2.imwrite(name, first_row)

def calangle(mask):
    '''
    计算图片旋转角度
    mask 二值化图片
    '''
    kernel = np.ones((1,3),np.uint8)
#closed = cv2.morphologyEx(mask ,cv2.MORPH_OPEN,kernel, iterations = 2)
#ofs.show(closed, 'On', 'closed')
    i = 0
    sure_bg = cv2.dilate(mask,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    for i in range(10):        
        sure_bg = cv2.dilate(unknown,kernel,iterations=3)
        dist_transform = cv2.distanceTransform(unknown,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)
    
    
    ofs.show(unknown, 'On', 'unknown')
    whitelinest = []
    for p in range(600, 700):
        i = 15
        for i in range(15 ,unknown.shape[0] - 1):
            if unknown[i, p] == 255:
                if unknown[i-1, p] == 0:
                    whitelinest.append(i)
                elif unknown[i+1, p] == 0:
                    whitelinest.append(i)
        if len(whitelinest) != 38:
                break
        else:
            whitelinest.clear()
    
    width = (unknown.shape[1] - p) + 1       
    
    i = 0
    whitelineend = []
    for i in range(0, unknown.shape[0] - 2):
        if unknown[i, unknown.shape[1] - 1] == 255:
            if unknown[i-1, unknown.shape[1] - 1] == 0:
                whitelineend.append(i)
            elif unknown[i+1, unknown.shape[1] - 1] == 0:
                whitelineend.append(i)
    length = (unknown.shape[0] - whitelineend[38]) + 1
    
    angle = math.atan(length/width) * 2 * math.pi
    print("angle = ", angle)
