import cv2
import numpy as np
import pandas as pd
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize,medial_axis
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage.measure import regionprops_table
import random as rng

rng.seed(12345)



def thresh_callback(val,gray):
    threshold = val
    # Detect edges using Canny
    canny_output = cv2.Canny(gray, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    # Show in a window
    cv2.imshow('Contours', drawing)
    
def active_neihbours(x,y,image):
    """Return 8-active_neighbours of image point P1(x,y), in a clockwise order"""
     
    img = image.copy()
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
    active_neighbours = []
    if img[x_1][y]!= 0 :
        active_neighbours.append([x_1,y])
    if img[x_1][y1] != 0:
        active_neighbours.append([x_1,y1])
    if img[x][y1] != 0:
        active_neighbours.append([x,y1])        
    if img[x1][y1] != 0:
        active_neighbours.append([x1,y1])
    if img[x1][y] != 0:
        active_neighbours.append([x1,y])
    if img[x1][y_1] != 0:
        active_neighbours.append([x1,y_1])
    if img[x][y_1] !=0:
        active_neighbours.append([x,y_1])
    if img[x_1][y_1] != 0:
        active_neighbours.append([x_1,y_1])
        
    return active_neighbours 
    

def neighbours(x,y,image):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1;
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]   

def image_resize(image, width = 800, height = 600, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def bin_to_bgr(image):
        """Transform the image to a ndarray with depth:3"""
        img = image.copy()
        h, w = img.shape
        image_bgr = np.zeros((h, w, 3))
        image_bgr[:, :, 0] = img
        image_bgr[:, :, 1] = img
        image_bgr[:, :, 2] = img
        imge= image_bgr
        return img
    
def bin_to_bgr_(bin_image):
    rgb_img = cv2.cvtColor(bin_image, cv2.COLOR_GRAY2RGB)
    return rgb_img

def get_uint_image(image):
    """
    Returns the np_image converted to uint8 and multiplied by 255 to simulate grayscale
    :return: a ndarray image
    """
    #image = np.uint8(image) *255
    img = image.astype(np.uint8) * 255
    return img

def thinning_zhang_suen(image):
    "the Zhang-Suen Thinning Algorithm"
    image_thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        # the points to be removed (set as 0)
    while changing1 or changing2:   # iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns,_ = image_thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(x, y, image_thinned)
                if ((image_thinned[x][y] == 1).all() and    # Condition 0: Point p1 in the object regions
                    2 <= sum(n) <= 6 and    # Condition 1: 2<= N(p1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(p1)=1
                    p2 * p4 * p6 == 0 and    # Condition 3
                    p4 * p6 * p8 == 0):         # Condition 4
                        changing1.append((x, y))
        for x, y in changing1:
            image_thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                p2, p3, p4, p5, p6, p7, p8, p9 = n = neighbours(x, y, image_thinned)
                if ((image_thinned[x][y] == 1).all() and        # Condition 0
                    2 <= sum(n) <= 6 and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    p2 * p4 * p8 == 0 and       # Condition 3
                    p2 * p6 * p8 == 0):            # Condition 4
                        changing2.append((x, y))
        for x, y in changing2:
            image_thinned[x][y] = 0
    return image_thinned

def skeleton(img):
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    
    ret,img = cv2.threshold(img,0,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    
    return skel

def edgesmoothing(thresh):
    blur = cv2.pyrUp(thresh)

    for i in range(15):
        blur = cv2.medianBlur(blur,5)

    blur = cv2.pyrDown(blur)
    ret,ths = cv2.threshold(blur,30,255,cv2.THRESH_BINARY)

    return ths

def denoise(thresh,thresh_area):
    edges = cv2.Canny(thresh,0,255)

    ret,threshs = cv2.threshold(edges,200,255,0)
    cnts, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    outer = thresh.copy()


    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) 
    rect_areas = []
    for c in cnts:   
        (x, y, w, h) = cv2.boundingRect(c)
        rect_areas.append(w * h)
    avg_area = np.mean(rect_areas)

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cnt_area = w * h
        if cnt_area<thresh_area:
            outer[y:y + h, x:x + w] = 0

    return outer

def border(image,size):
    cur = image.shape[:2]
    dw = size - cur[1]
    dh = size - cur[0]

    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
    
    x = cv2.resize(new_im,(size,size))

    return x


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print('Error loading image')
        exit()  
    return img
def save_image(path, image):
    cv2.imwrite(path, image)