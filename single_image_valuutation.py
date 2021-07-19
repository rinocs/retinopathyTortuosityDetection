import cv2
import numpy as np
from skimage import color, feature, filters, io
from skimage.morphology import skeletonize,medial_axis
from skimage.measure import regionprops_table, regionprops, label
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


trainSplittedPath ='sample/CHASE/train/label/splitted_vessels/'
testSplittedPath ='sample/CHASE/test/label/splitted_vessels/'

validateSplittedPath = 'sample/CHASE/validate/label/splitted_vessels/'




filename = "vImage_05R_1stHO_21.png"

img = image_util.load_image(os.path.join(trainSplittedPath, filename))  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY )
binary = thresh > filters.threshold_otsu(thresh)
np.unique(binary)
##### SKELETON 
# skel, distance = medial_axis(binary, return_distance=True)

skeleton = skeletonize(binary)
# Distance to the background for pixels of the skeleton
# dist_on_skel = distance * skel
skeleton = image_util.get_uint_image(skeleton)

# cv2.imshow("skel", skeleton)
# cv2.waitKey(0)

output = cv2.connectedComponentsWithStats(skeleton,8, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output
# vessels_util.imshow_components(labels)
# props = regionprops_table(labels,None,('label','coords','bbox','area','centroid'))
for region  in regionprops(labels):
    coord = region['coords']
    x_coords = []
    y_coords = []
    for p in region['coords']:
        x_coords.append(p[0])
        y_coords.append(p[1])
    distance_tort = tortuosity.distance_measure_tortuosity(coord)
    curve_length = tortuosity._curve_length(coord)
    chord_length = tortuosity._chord_length(coord)
    #compute mean standard deviation of angels between lines tangent to each pixel along centerline and a reference axis
    sd, slope = tortuosity.sd_theta(x_coords, y_coords) 
    #compute mean distance measure (ratio of actual length to chord length) between inflection points
    mean_dm,num_inflection_pts,curvature,DM = tortuosity.mean_distance_measure(x_coords, y_coords)
    
    # compute number of critical points
    num_cpts = tortuosity.num_critical_points(x_coords, y_coords)

    #compute VTI index
    VTI = tortuosity.vessel_tort_index(curve_length,sd,num_cpts,mean_dm,chord_length)
    
print('image:', filename )
print("coords: ", coord)
print('curve_length:', curve_length)
print('chord_length: ', chord_length)
print('sd_theta:', sd)
print('num_inflection_pts: ', num_inflection_pts)
print('num_critical_points: ',num_cpts)
print('curvature: ', curvature)
print('VTI:',VTI)
print('distance_tort:', distance_tort)