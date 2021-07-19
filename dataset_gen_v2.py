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

    
# with open(trainSplittedPath+'train_splitted.csv', mode='w+') as csv_file:
#     fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts','num_critical_points','curvature','VTI','distance_tort']
#     writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

#     writer.writeheader()

    
        
#     for filename in os.listdir(trainSplittedPath):
       
#         if filename.endswith(".png") : 
#             # print(os.path.join(trainPath, filename))
#             # continue
#             img = image_util.load_image(os.path.join(trainSplittedPath, filename))  
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#             ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY )
#             binary = thresh > filters.threshold_otsu(thresh)
#             np.unique(binary)
#             ##### SKELETON 
#             # skel, distance = medial_axis(binary, return_distance=True)

#             skeleton = skeletonize(binary)
#             # Distance to the background for pixels of the skeleton
#             # dist_on_skel = distance * skel
#             skeleton = image_util.get_uint_image(skeleton)
            
#             # cv2.imshow("skel", skeleton)
#             # cv2.waitKey(0)
         
#             output = cv2.connectedComponentsWithStats(skeleton,8, cv2.CV_32S)
#             (numLabels, labels, stats, centroids) = output
#             # vessels_util.imshow_components(labels)
#             # props = regionprops_table(labels,None,('label','coords','bbox','area','centroid'))
#             for region  in regionprops(labels):
#                 coord = region['coords']
#                 x_coords = []
#                 y_coords = []
#                 for p in region['coords']:
#                     x_coords.append(p[0])
#                     y_coords.append(p[1])
#                 distance_tort = tortuosity.distance_measure_tortuosity(coord)
#                 curve_length = tortuosity._curve_length(coord)
#                 chord_length = tortuosity._chord_length(coord)
#                 #compute mean standard deviation of angels between lines tangent to each pixel along centerline and a reference axis
#                 sd, slope = tortuosity.sd_theta(x_coords, y_coords) 
#                 #compute mean distance measure (ratio of actual length to chord length) between inflection points
#                 mean_dm,num_inflection_pts,curvature,DM = tortuosity.mean_distance_measure(x_coords, y_coords)
                
#                 # compute number of critical points
#                 num_cpts = tortuosity.num_critical_points(x_coords, y_coords)

#                 #compute VTI index
#                 VTI = tortuosity.vessel_tort_index(curve_length,sd,num_cpts,mean_dm,chord_length)
#             writer.writerow({'image': filename, 'curve_length': curve_length, 'chord_length': chord_length,'sd_theta': sd,'num_inflection_pts': num_inflection_pts,'num_critical_points':num_cpts,'curvature': curvature,'VTI':VTI,'distance_tort': distance_tort})
# csv_file.close()            
                
            
    
       

with open(testSplittedPath+'test_splitted.csv', mode='w+') as csv_file1:
    fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts','num_critical_points','curvature','VTI','distance_tort']
    writer1 = csv.DictWriter(csv_file1, fieldnames=fieldnames)

    writer1.writeheader()

    
        
    for filename in os.listdir(testSplittedPath):
       
        if filename.endswith(".png") : 
#            # print(os.path.join(testSplittedPath, filename))
            # continue
            img = image_util.load_image(os.path.join(testSplittedPath, filename))  
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
            writer1.writerow({'image': filename, 'curve_length': curve_length, 'chord_length': chord_length,'sd_theta': sd,'num_inflection_pts': num_inflection_pts,'num_critical_points':num_cpts,'curvature': curvature,'VTI':VTI,'distance_tort': distance_tort})
          
csv_file1.close()  
       
with open(validateSplittedPath+'validate_splitted.csv', mode='w+') as csv_file2:
    fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts','num_critical_points','curvature','VTI','distance_tort']
    writer2 = csv.DictWriter(csv_file2, fieldnames=fieldnames)

    writer2.writeheader()

    
        
    for filename in os.listdir(validateSplittedPath):
       
        if filename.endswith(".png") : 
            # print(os.path.join(validateSplittedPath, filename))
            # continue
            img = image_util.load_image(os.path.join(validateSplittedPath, filename))  
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
            writer2.writerow({'image': filename, 'curve_length': curve_length, 'chord_length': chord_length,'sd_theta': sd,'num_inflection_pts': num_inflection_pts,'num_critical_points':num_cpts,'curvature': curvature,'VTI':VTI,'distance_tort': distance_tort})
csv_file2.close()      
        