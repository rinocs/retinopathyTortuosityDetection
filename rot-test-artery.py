import cv2
import numpy as np
import pandas as pd
import json
from utils.math import tortuosity
import scipy.io
import os
import time

# import sys
# print(sys.maxsize)
from matplotlib import pyplot as plt

#  print(os.path.abspath(os.curdir)) check current dir
mat_file = scipy.io.loadmat('./Data/tortData.mat')

seg_art = mat_file['seg_art']
# function to add to JSON
def write_json(data, filename='./Data/artery_tort_test_1.json'):
    with open(filename,'w') as f:
        json.dump(data, f, indent=4)
      
      
data_json = {}

vti_matlab = []
file = open("./Data/art_tort_vti.txt").readlines()
for lines in file:
    vti_matlab.append(lines)



for i in range(0,len(seg_art[0])-1):
    name = seg_art[0][i][0]
    rank = seg_art[0][i][3]
    x_coords = seg_art[0][i][1].tolist()
    y_coords = seg_art[0][i][2].tolist()
    coords = [[x,y] for x,y in zip(x_coords[0],y_coords[0])]
    all_tort = 0
    #coords_array = np.array([coord], dtype=np.int32)
    curve_length = tortuosity._curve_length(coords)
    chord_length = tortuosity._chord_length(coords)
    #compute mean standard deviation of angels between lines tangent to each pixel along centerline and a reference axis
    sd, slope = tortuosity.sd_theta(x_coords[0], y_coords[0]) 
     #compute mean distance measure (ratio of actual length to chord length) between inflection points
    mean_dm,num_inflection_pts,curvature,DM = tortuosity.mean_distance_measure(x_coords[0], y_coords[0])
    
    # compute number of critical points
    num_cpts = tortuosity.num_critical_points(x_coords[0], y_coords[0])

    #compute VTI index
    VTI = tortuosity.vessel_tort_index(curve_length,sd,num_cpts,mean_dm,chord_length)
    
    tortuosity_measure = tortuosity.distance_measure_tortuosity(coords)
    linear_reg_tort = tortuosity.linear_regression_tortuosity(x_coords[0], y_coords[0], 50)
    squared_tort = tortuosity.squared_curvature_tortuosity(x_coords[0],y_coords[0])
    distance_inflection_tort = tortuosity.distance_inflection_count_tortuosity(x_coords[0], y_coords[0])
    tort_density = tortuosity.tortuosity_density(x_coords[0],y_coords[0])
    #curve_img = tortuosity._curve_to_image(x_coords[0], y_coords[0])
    #print(np.any(curve_img[:, :] == 1 ))
    #plt.imshow(curve_img, cmap=plt.cm.gray)

    #time.sleep(3)
    
    
    


    # python object to be appended
    y = { str(name): {
    "chord": chord_length,
    "arch": curve_length,
    "sd": sd,
    "slope": list(slope),
    "mean_dm": mean_dm,
    "num_inflection_points": num_inflection_pts,
    "num_critical_points": num_cpts,
    "curvature": curvature,
    "distance_tortuosity": tortuosity_measure,
    "linear_reg_tort": linear_reg_tort,
    "squared_tort": squared_tort,
    "distance_inflection_tort": distance_inflection_tort,
    "tortuosity_density": tort_density,
    "vti": VTI,
    "vti_mat": float(vti_matlab[i]),
    "rank": int(rank)
    }}
    


    # appending data to artery_measurements 
    data_json.update(y)
        

write_json(data_json) 



