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
mat_file = scipy.io.loadmat('./myProject/Data/tortData.mat')

seg_vei = mat_file['seg_vei']
# function to add to JSON
def write_json(data, filename='./myProject/Data/veins_tort.json'):
    with open(filename,'w') as f:
        json.dump(data, f, indent=4)
      
      
data_json = {}


for i in range(0,len(seg_vei[0])-1):
    name = seg_vei[0][i][0]
    rank = seg_vei[0][i][3]
    x_coords = seg_vei[0][i][1].tolist()
    y_coords = seg_vei[0][i][2].tolist()
    coords = [[x,y] for x,y in zip(x_coords[0],y_coords[0])]
    all_tort = 0
    #coords_array = np.array([coord], dtype=np.int32)
    curve_length = tortuosity._curve_length(coords)
    chord_length = tortuosity._chord_length(coords)
    tortuosity_measure = tortuosity.distance_measure_tortuosity(coords)
    linear_reg_tort = tortuosity.linear_regression_tortuosity(x_coords[0], y_coords[0], 50)
    squared_tort = tortuosity.squared_curvature_tortuosity(x_coords[0],y_coords[0])
    distance_inflection_tort = tortuosity.distance_inflection_count_tortuosity(x_coords[0], y_coords[0])
    tort_density = tortuosity.tortuosity_density(x_coords[0],y_coords[0])
    curve_img = tortuosity._curve_to_image(x_coords[0], y_coords[0])
    spline = tortuosity.smooth_tortuosity_cubic(x_coords[0], y_coords[0], name)
    
    print(spline)
    #print(np.any(curve_img[:, :] == 1 ))
    # plt.imshow(curve_img, cmap=plt.cm.gray)

    #time.sleep(3)
    
    
    


    # python object to be appended
    y = { str(name): {
    "chord": chord_length,
    "arch": curve_length,
    "distance_tortuosity": tortuosity_measure,
    "linear_reg_tort": linear_reg_tort,
    "squared_tort": squared_tort,
    "distance_inflection_tort": distance_inflection_tort,
    "tortuosity_density": tort_density,
    "rank": int(rank)
    }}
    


    # appending data to artery_measurements 
    data_json.update(y)
        

write_json(data_json) 



