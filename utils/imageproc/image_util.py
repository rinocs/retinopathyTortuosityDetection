import cv2
import numpy as np
import pandas as pd
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize,medial_axis
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from skimage.measure import regionprops_table

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