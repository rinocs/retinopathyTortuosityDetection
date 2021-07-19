import numpy as np
import os
import cv2
import pandas as pd
import h5py
import logging 
import matplotlib.pyplot as plt
import seaborn as sns



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.models import dataprepare, models


#### Path 

trainSplittedPath ='sample/CHASE/train/label/splitted_vessels/'
testSplittedPath ='sample/CHASE/test/label/splitted_vessels/'
validateSplittedPath = 'sample/CHASE/validate/label/splitted_vessels/'

### parameters
image_size = 224
h,w = 85,98

### load data
logging.info("loading data")
print("loading data")

# x_train, y_train = dataprepare.read_and_normalize_train_data_v1(trainSplittedPath, image_size)
# x_test, y_test = dataprepare.read_and_normalize_test_data_v1(testSplittedPath, image_size)
# x_val, y_val = dataprepare.read_and_normalize_val_data_v1(validateSplittedPath, image_size)
# x_train, y_train = dataprepare.read_and_normalize_train_data_v2(trainSplittedPath, image_size)
# x_test, y_test = dataprepare.read_and_normalize_test_data_v2(testSplittedPath, image_size)
# x_val, y_val = dataprepare.read_and_normalize_val_data_v2(validateSplittedPath, image_size)
x_train, y_train = dataprepare.read_train_data_v2(trainSplittedPath,  h,w)
x_test, y_test = dataprepare.read_test_data_v2(testSplittedPath,  h,w)
x_val, y_val = dataprepare.read_val_data_v2(validateSplittedPath,  h,w)
# x_train = x_train.reshape(-1, image_size, image_size, 1)
# x_val = x_val.reshape(-1, image_size, image_size, 1)
# x_test = x_test.reshape(-1, image_size, image_size, 1)

np.save('x_train_'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', x_train)
np.save('y_train_'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', y_train)
np.save('x_test_'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', x_test)
np.save('y_test_'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', y_test)
np.save('x_val_'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', x_val)
np.save('y_val_'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', y_val)

print("data loaded and matrix saved")