import logging
import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adadelta, Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils.models import dataprepare, models

#### Path

rot_path = "sample/tort/skeletons/"

### parameters
image_size = 224
h, w = 85, 98
h1, w1 = 53, 427

### load data
logging.info("loading data")
print("loading data")

# x_train, y_train = dataprepare.read_and_normalize_train_data_v1(trainSplittedPath, image_size)
# x_test, y_test = dataprepare.read_and_normalize_test_data_v1(testSplittedPath, image_size)
# x_val, y_val = dataprepare.read_and_normalize_val_data_v1(validateSplittedPath, image_size)
# x_train, y_train = dataprepare.read_and_normalize_train_data_v2(trainSplittedPath, image_size)
# x_test, y_test = dataprepare.read_and_normalize_test_data_v2(testSplittedPath, image_size)
# x_val, y_val = dataprepare.read_and_normalize_val_data_v2(validateSplittedPath, image_size)
# x_train, y_train = dataprepare.read_tort_data_dt(rot_path,  h1,w1)
x_train, y_train = dataprepare.read_tort_data(rot_path, h1, w1)

# x_train = x_train.reshape(-1, image_size, image_size, 1)
# x_val = x_val.reshape(-1, image_size, image_size, 1)
# x_test = x_test.reshape(-1, image_size, image_size, 1)

np.save("x_train_" + str(h1) + "_" + str(w1) + "tort_rank.npy", x_train)
np.save("y_train_" + str(h1) + "_" + str(w1) + "tort_rank.npy", y_train)


print("data loaded and matrix saved")
