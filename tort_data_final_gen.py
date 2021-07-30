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
import scipy.io

mat_file = scipy.io.loadmat('../myProject/Data/tortData.mat')
skeletonPath = "sample/tort/skeletons/"

arteria = 'Arteria'
vena = 'Vena'
seg_vei = mat_file['seg_vei']
seg_art = mat_file['seg_art']
# seg_all = seg_vei + seg_art

with open(skeletonPath+'all_tort2.csv', mode='w+') as csv_file:
    fieldnames = ['image', 'curve_length', 'chord_length','sd_theta','num_inflection_pts',
                    'num_critical_points','curvature','VTI','distance_tort',
                    'mat_curve_length',
                    'mat_chord_length',
                    'mat_sd_theta',
                    'mat_num_inflection_pts',
                    'mat_num_critical_points',
                    'mat_curvature',
                    'mat_VTI' ,
                    'mat_distance_tort',    
                    'rank']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for filename in os.listdir(skeletonPath):
        rank = 0
        if filename.endswith(".png") : 
            # print(os.path.join(trainPath, filename))
            # continue
            mat_name = filename.split('_')[0]
            gender = mat_name.split('-')[0]
            num = mat_name.split('-')[1]
            correct_name = num+'_'+gender+'.jpg'
            if gender == arteria:
                for i in range (0, len(seg_art[0])-1):
                
                    if correct_name == seg_art[0][i][0]:
                        rank = int(seg_art[0][i][3]) 
                        mat_x_coords = seg_art[0][i][1].tolist()
                        mat_y_coords = seg_art[0][i][2].tolist()
            else:
                for i in range(0, len(seg_vei[0])-1):
                    if correct_name == seg_vei[0][i][0]:
                        rank = int(seg_vei[0][i][3])    
                        mat_x_coords = seg_vei[0][i][1].tolist()
                        mat_y_coords = seg_vei[0][i][2].tolist() 
            mat_coords = [[x,y] for x,y in zip(mat_x_coords[0],mat_y_coords[0])]
            img = image_util.load_image(os.path.join(skeletonPath, filename))  
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
         
            # output = cv2.connectedComponentsWithStats(skeleton,4, cv2.CV_32S)
            # (numLabels, labels, stats, centroids) = output
            labels, numLabels = label(skeleton,return_num=True)
            # plt.imshow(labels)
            # plt.show()
            print(numLabels)
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
            
            mat_distance_tort = tortuosity.distance_measure_tortuosity(mat_coords)
            mat_curve_length = tortuosity._curve_length(mat_coords)
            mat_chord_length = tortuosity._chord_length(mat_coords)
            #compute mean standard deviation of angels between lines tangent to each pixel along centerline and a reference axis
            mat_sd, mat_slope = tortuosity.sd_theta(mat_x_coords[0], mat_y_coords[0]) 
            #compute mean distance measure (ratio of actual length to chord length) between inflection points
            mat_mean_dm,mat_num_inflection_pts,mat_curvature,mat_DM = tortuosity.mean_distance_measure(mat_x_coords[0], mat_y_coords[0])
            
            # compute number of critical points
            mat_num_cpts = tortuosity.num_critical_points(mat_x_coords[0], mat_y_coords[0])

            #compute VTI index
            mat_VTI = tortuosity.vessel_tort_index(mat_curve_length,mat_sd,mat_num_cpts,mat_mean_dm,mat_chord_length)
            writer.writerow({'image': filename, 'curve_length': curve_length,
                             'chord_length': chord_length,'sd_theta': sd,'num_inflection_pts': num_inflection_pts,
                             'num_critical_points':num_cpts,'curvature': curvature,
                             'VTI':VTI,'distance_tort': distance_tort,
                             'mat_curve_length': mat_curve_length,
                             'mat_chord_length': mat_chord_length,
                             'mat_sd_theta': mat_sd,
                             'mat_num_inflection_pts': mat_num_inflection_pts,
                             'mat_num_critical_points': mat_num_cpts,
                             'mat_curvature': mat_curvature,
                             'mat_VTI': mat_VTI ,
                             'mat_distance_tort': mat_distance_tort,
                             'rank': rank})
csv_file.close()   