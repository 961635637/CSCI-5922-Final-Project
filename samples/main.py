import cv2
import glob
import os
import shutil
import numpy as np



from balloon import load_weights_new
from balloon import read_one
from balloon import detect_and_color_splash
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import model as modellib, utils

top_path = '/home/yan/mask_RCNN/samples/balloon/total'

save_top_path = '/home/yan/mask_RCNN/samples/balloon/Results'

weight_path = "/home/yan/mask_RCNN/logs/corn20181213T1042/mask_rcnn_corn_0102.h5"
log_path = "/home/yan/mask_RCNN/logs/"


def get_paths(top_path, n_classes = 1, image_type = '*.jpg'):

    #Full poths
    total = 0 # The total numnber of images
    roots = []
    img_paths = []
    label_real_names = []
    
    for root, dirs, files in os.walk(top_path):
        roots.append(os.path.join(root, image_type))
        for dirc in dirs:
            label_real_names.append(dirc)
        for file in files:
            total = total + 1
    del roots[0]  # delete the root directory
    #print(roots)
    print("\nThe total number of full images is " + str(total) + ' .')
    print("The type name of full images are " + str(label_real_names))
    try:
        assert n_classes == len(roots)
    except AssertionError:
        raise AssertionError("[-] Hi, man! How many types in full image folder do you have?")
       
       
    for i in range(0, len(roots)):
        img_paths.append(glob.glob(roots[i]))

    print(sum(img_paths, []))
    return sum(img_paths, []), label_real_names

def create_dir(save_top_path, label_real_names):
    if os.path.isdir(save_top_path):
        shutil.rmtree(save_top_path)
    os.mkdir(save_top_path)
    
    for label in label_real_names:
        label_path = save_top_path + '/' + label
        if not os.path.exists(label_path):
             os.mkdir(label_path)

img_paths, label_real_names = get_paths(top_path, n_classes = 1, image_type = '*.jpg')
create_dir(save_top_path, label_real_names)

def main(img_paths, save_top_path, weightsp = weight_path, logsp = log_path):
    model = load_weights_new()
    for img_path in img_paths:
        print(img_path.split('/')[-1])
        save_path = save_top_path + '/'+ img_path.split('/')[-2]+ '/' +  img_path.split('/')[-1]
        print(save_path)
        #read_resize(img_path, save_path, size)
        #read_one(imagep = img_path, savep = save_top_path +'/crop_rgb/')
        #model = load_weights_new()
        detect_and_color_splash(model, image_path=img_path, save_path = save_top_path +'/crop_rgb/')
        os.remove(img_path)


main(img_paths, save_top_path)
