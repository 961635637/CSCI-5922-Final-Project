"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import random
import itertools
import colorsys
import math
import cv2
from scipy import ndimage
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches, lines

from matplotlib.patches import Polygon
import IPython.display

from scipy import stats

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
font = cv2.FONT_HERSHEY_SIMPLEX

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils



############################################################
#  Visualization
############################################################

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


def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def rotate(xval, yval, gradient):
    if gradient <=0 :
        sintheta = math.sin(math.atan(gradient))
        costheta = math.cos(math.atan(gradient))
        mulmatrix = np.array([[costheta, sintheta],[-sintheta, costheta]])
    else:
        sintheta = math.sin(math.atan(gradient)-(math.pi/2))
        costheta = math.cos(math.atan(gradient)-(math.pi/2))
        mulmatrix = np.array([[costheta, sintheta],[-sintheta, costheta]])
    inputvec = np.array([xval, yval])
    return np.dot(mulmatrix, inputvec)

def rotate_reverse(xval, yval, gradient):
    if gradient <= 0:
        sintheta = math.sin(math.atan(gradient))
        costheta = math.cos(math.atan(gradient))
        mulmatrix = np.array([[costheta, -sintheta],[sintheta, costheta]])
    else:
        sintheta = math.sin(math.atan(gradient)-(math.pi/2))
        costheta = math.cos(math.atan(gradient)-(math.pi/2))
        mulmatrix = np.array([[costheta, -sintheta],[sintheta, costheta]])
    inputvec = np.array([xval, yval])
    return np.dot(mulmatrix, inputvec)

#def display_instances_1()
    
def display_instances(image, image_path, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optionamin_yvall) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    xval = []
    yval = []
    xcornor = []
    ycornor = []
    rotatebool = False
    N = boxes.shape[0]
    name = image_path
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #print(masks.shape)
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        automin_yval_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('on')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    bitmask = new_bina(image)
    #print(masks.shape)
    for i in range(N):
        if scores[i] < 0.95:
            continue;
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        #print(y1, x1, y2, x2)
        show_bbox = False
        for h in range(y1, y2):
            for w in range(x1, x2):
                if masks[h, w, i] == True:
                    xval.append(w)
                    yval.append(h)
        
        gradient, intercept, r_value, p_value, std_err = stats.linregress(xval, yval)
        #print(gradient)
        #if gradient >0:
            #f.open('bad.txt', 'a')
            #for line:
            #save image_path
        if gradient <=0 :
            xval2 = rotate(xval, yval, gradient)[0]
            yval2 = rotate(xval, yval, gradient)[1]
            min_xval = min(xval2)
            max_xval = max(xval2)
            min_yval = min(yval2)
            max_yval = max(yval2)
            #plt.plot(xval, yval,'ob')
            #plt.plot(xval2, yval2,'ob')
            count, xpoints, ypoints, rate = getWidth(min_yval, max_yval, min_xval, max_xval, image, gradient, xval, yval, xval2, yval2, bitmask)
            #print(rate)
            xpoints = rotate_reverse(xpoints, ypoints, gradient)[0]
            ypoints = rotate_reverse(xpoints, ypoints, gradient)[1]
            #print('!!!!!!!!!!!!!!!!!!!!!!!!!!111111')
            #print((xpoints[0]))
            #print((ypoints[0]))
            if len(xpoints) != 0 and len(ypoints) != 0:
                cv2.putText(image,str(round(rate,2)),(3, int(ypoints[0])), font, 0.5,(0,0,255),1,cv2.LINE_AA)
            for i in range(0, len(xpoints)-1,2):
                cv2.rectangle(image, (int(xpoints[i]), int(ypoints[i])), (int(xpoints[i+1]), int(ypoints[i+1])), (0, 0, 255), 1)
                #show(image, 'on', '_rotate')
            xval = []
            yval = []
            xval2 = []
            yval2 = []
            rotatebool = False
        elif gradient > 0:
            new_img, name = rotate_image(image, image_path, -90, save = 'on')
            rotatebool = True
        #plt.plot([xval3[0], xval3[2]], [yval3[0], yval3[2]], color = 'red')
        #plt.plot([xval3[1], xval3[3]], [yval3[1], yval3[3]], color = 'red')
        #plt.plot([xval3[0], xval3[1]], [yval3[0], yval3[1]], color = 'red')
        #plt.plot([xval3[2], xval3[3]], [yval3[2], yval3[3]], color = 'red')
         
        
        
        #ax.add_patch(p)
        # Mask
        #mask = masks[:, :, i]
    
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.

    return name, rotatebool





def getWidth(up, down, start, end, image, gradient, xval, yval, xval2, yval2, bitmask):
    '''
    mask 二值化， temp 红蓝线列表
    返回苗的宽度
    '''
#求每行有多少苗
    
    count = 0
    label = False
    Blackpix = 0
    Whitepix = 0
    label2 = False
    
    xpoints = []
    ypoints = []


    '''
    hsv_green=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    H, S, V = cv2.split(hsv_green)
    LowerGreen = np.array([45, 70, 20])  #51, 42, 15
    UpperGreen = np.array([75, 255, 255]) #191 255 0
    bitmask = cv2.inRange(hsv_green, LowerGreen, UpperGreen)
    #GreenThings = cv2.bitwise_and(hsv_green, hsv_green, mask=bitmask)
    '''
    
    #cv2.imwrite('/home/yan/mask_RCNN/samples/balloon/crop_mask.jpg', bitmask)
    
    RGB=[0]*len(xval)
    for i in range(len(xval)):
        RGB[i] = bitmask[yval[i]][xval[i]]
    
    #maxxval = int(max(xval2))
    #maxyval = int(max(yval2))
    new_mask = np.zeros((int(down)+1, int(end)+1))
    for i in range(0, len(xval2)):
        #print(int(xval2[i]))
        new_mask[int(yval2[i])][int(xval2[i])] = RGB[i]
    #'''
    #startline = True
    blackcol = []
    for q in range(int(start), int(end) + 1): 
        #sum = 0
        blackline = True
        for p in range(int(up), int(down) + 1): 
            if new_mask[p][q] == 255:
                blackline = False
        if blackline == True:
            blackcol.append(q)
            
    startline = blackcol[0]
    area = 0
    for i in range(1, len(blackcol)):
        if blackcol[i] == blackcol[i-1]+1:
            startline = startline
            
        elif blackcol[i] != blackcol[i-1]+1:
            if (blackcol[i-1] - startline) > 10:
                xpoints.append(startline)
                ypoints.append(int(down))
                xpoints.append(blackcol[i-1])
                ypoints.append(int(up))
                area = area + blackcol[i-1] - startline
            startline = blackcol[i]
    rate = area/(end-start)
               
                
    return count, xpoints, ypoints, 1-rate


def show(image, save = 'off', name = 'show_temp'):
    """
    显示并存储图片。如果save为on，则存储图片，否则仅显示。'/home/group/mask_RCNN/datasets/corn/val/'
    name为保存的图片名
    
    plt.figure()
    plt.title('The ' + name + ' image')
    plt.imshow(image)
    plt.show()
    """
    
    if save == 'on' or 'ON' or 'On' or 'oN':
        name = name + '.jpg'
        cv2.imwrite(name, image)

def rotate_image(image, image_path, angle, save = 'off'):
    """
    将照片旋转多少度（angle）
    image是原照片，display是否显示，save是否保存，name旋转后的照片名字
    """
    #rotation angle in degree
    res = ndimage.rotate(image, angle)
    

    if save == 'on' or 'ON' or 'On' or 'oN':        
        name =  image_path.split('.')[0] + '_rotate.jpg'
        cv2.imwrite(name, res)
    return res, name



def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=True, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""
    # Match predictions to ground truth
    gt_match, pred_match, overlaps = utils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)
    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0, .8)] * len(gt_match)\
           + [(1, 0, 0, 1)] * len(pred_match)
    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)
    # Captions per instance show score/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
            for i in range(len(pred_match))]
    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)


def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visumatplotlib linesalization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictions and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with different
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominent each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            x = random.randint(x1, (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)
