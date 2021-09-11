import os 
import sys 
import cv2 
import numpy as np
from utils.imageproc import image_util



### path

rot_path = 'sample/tort/skeletons/'
new_rot_path = 'sample/tort/skeletons_new/'

size = 850



for filename in os.listdir(rot_path):
       
    if filename.endswith(".png"): 
        
        img = image_util.load_image(os.path.join(rot_path, filename)) 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # cv2.imshow("original", img)
        # cv2.waitKey(0)   
        #get size
        height, width = img.shape
        print (filename,height, width)
        # Create a black image
        x = height if height > width else width
        y = height if height > width else width
        square= np.zeros((size,size), np.uint8)
        #
        #This does the job
        #
        # square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
        square[int((y-height)/2):int(y-(y-height)/2), int((x-width)/2):int(x-(x-width)/2)] = img
        cv2.imwrite(new_rot_path+filename,square)
        # cv2.imshow("original", img)
        # cv2.imshow("black square", square)
        # cv2.waitKey(0)   