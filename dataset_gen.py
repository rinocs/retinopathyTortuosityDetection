import cv2
import numpy as np
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize,medial_axis
import sys
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from utils.imageproc import image_util
from utils.vessels import vessels_util
from utils.math import tortuosity
import math
import csv
import os 
import logging
from progress.bar import Bar
bar = Bar('Processing', max=100)
trainPath  = 'sample/CHASE/train/label/'
trainSplittedPath ='sample/CHASE/train/label/splitted_vessels/'
testSplittedPath ='sample/CHASE/test/label/splitted_vessels/'
testPath  = 'sample/CHASE/test/label/'
validatePath = 'sample/CHASE/validate/label/'
validateSplittedPath = 'sample/CHASE/validate/label/splitted_vessels/'

    
# with open(trainSplittedPath+'train_splitted.csv', mode='w+') as csv_file:
#     fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts','num_critical_points','curvature','VTI','distance_tort']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#     writer.writeheader()

    
        
#     for filename in os.listdir(trainPath):
       
#         if filename.endswith(".png") : 
#             # print(os.path.join(trainPath, filename))
#             # continue
#             img = image_util.load_image(os.path.join(trainPath, filename))
#             resized = image_util.image_resize(img)   
#             gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
#             ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY )
#             binary = thresh > filters.threshold_otsu(thresh)
#             np.unique(binary)
#             ##### SKELETON 
#             # skel, distance = medial_axis(binary, return_distance=True)

#             skeleton = skeletonize(binary)
#             # Distance to the background for pixels of the skeleton
#             # dist_on_skel = distance * skel
#             skeleton = image_util.get_uint_image(skeleton)
#             branch_locations = vessels_util.getIntersections(skeleton)
#             labels = vessels_util.connected_component_label(skeleton.copy(), branch_locations)
#             props = vessels_util.separate_labels(labels,filename, trainSplittedPath,skeleton, writer)
#             bar.next()
#     bar.finish()
       

with open(testSplittedPath+'test_splitted.csv', mode='w+') as csv_file:
    fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts','num_critical_points','curvature','VTI','distance_tort']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    
        
    for filename in os.listdir(testPath):
       
        if filename.endswith(".png") : 
            # print(os.path.join(trainPath, filename))
            # continue
            img = image_util.load_image(os.path.join(testPath, filename))
            resized = image_util.image_resize(img)   
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
            ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY )
            binary = thresh > filters.threshold_otsu(thresh)
            np.unique(binary)
            ##### SKELETON 
            # skel, distance = medial_axis(binary, return_distance=True)

            skeleton = skeletonize(binary)
            # Distance to the background for pixels of the skeleton
            # dist_on_skel = distance * skel
            skeleton = image_util.get_uint_image(skeleton)
            branch_locations = vessels_util.getIntersections(skeleton)
            labels = vessels_util.connected_component_label(skeleton.copy(), branch_locations)
            props = vessels_util.separate_labels(labels,filename, testSplittedPath,skeleton, writer)
            bar.next()
    bar.finish()
       
with open(validateSplittedPath+'validate_splitted.csv', mode='w+') as csv_file:
    fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts','num_critical_points','curvature','VTI','distance_tort']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

    
        
    for filename in os.listdir(validatePath):
       
        if filename.endswith(".png") : 
            # print(os.path.join(trainPath, filename))
            # continue
            img = image_util.load_image(os.path.join(validatePath, filename))
            resized = image_util.image_resize(img)   
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) 
            ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY )
            binary = thresh > filters.threshold_otsu(thresh)
            np.unique(binary)
            ##### SKELETON 
            # skel, distance = medial_axis(binary, return_distance=True)

            skeleton = skeletonize(binary)
            # Distance to the background for pixels of the skeleton
            # dist_on_skel = distance * skel
            skeleton = image_util.get_uint_image(skeleton)
            branch_locations = vessels_util.getIntersections(skeleton)
            labels = vessels_util.connected_component_label(skeleton.copy(), branch_locations)
            props = vessels_util.separate_labels(labels,filename, validateSplittedPath,skeleton, writer)
            bar.next()
    bar.finish()
       