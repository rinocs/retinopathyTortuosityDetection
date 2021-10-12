import pandas
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

def load_val(path, h,w):
    X_val = []
    y_val = []
    print('Read tort images')
    for index, row in val.iterrows():
        image_path = os.path.join(path, str(row['image']))
        img = cv2.resize(cv2.imread(image_path, 0), (w, h) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        # img = img/255
        X_val.append(img)
        # y_val.append( [ row['VTI'] ] )
        # y_val.append( [ row['mat_distance_tort'] ] )
        y_val.append( [ row['distance_tort'] ] )
    return X_val, y_val
def read_tort_val_data(path,  h,w):
    val_data, val_target = load_val(path,  h,w)
    val_data = np.array(val_data, dtype=np.float32)
    val_target = np.array(val_target, dtype=np.float32)
    # val_data = val_data/255
    print('tort val shape:', val_data.shape)
    print(val_data.shape[0], 'tort val samples')
    return val_data, val_target

def load_test(path, h,w):
    X_test = []
    y_test = []
    print('Read tort images')
    for index, row in test.iterrows():
        image_path = os.path.join(path, str(row['image']))
        img = cv2.resize(cv2.imread(image_path, 0), (w, h) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        # img = img/255
        X_test.append(img)
        # y_test.append( [ row['mat_distance_tort'] ] )
        y_test.append( [ row['distance_tort'] ] )
    return X_test, y_test
def read_tort_test_data(path,  h,w):
    test_data, test_target = load_test(path,  h,w)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.float32)
    # val_data = val_data/255
    print('tort test shape:', test_data.shape)
    print(test_data.shape[0], 'tort val samples')
    return test_data, test_target

def load_train(path, h,w):
    X_train = []
    y_train = []
    print('Read tort images')
    for index, row in train.iterrows():
        image_path = os.path.join(path, str(row['image']))
        img = cv2.resize(cv2.imread(image_path, 0), (w, h) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        # img = img/255
        X_train.append(img)
        # y_test.append( [ row['mat_distance_tort'] ] )
        y_train.append( [ row['distance_tort'] ] )
    return X_train, y_train
def read_tort_train_data(path,  h,w):
    train_data, train_target = load_train(path,  h,w)
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    # val_data = val_data/255
    print('tort train shape:', train_data.shape)
    print(train_data.shape[0], 'tort train samples')
    return train_data, train_target



ALL_PATH = "sample/allChase/"
### parameters
image_size = 224
h,w = 85,98

df = pd.read_csv (ALL_PATH+'allcleaned.csv')

scaler = MinMaxScaler()
df["distance_tort"] = scaler.fit_transform(df['distance_tort'].values.reshape(-1,1))
train, val = train_test_split(df, test_size=0.2)
train, test = train_test_split(train, test_size=0.15)
# train["rank"] = scaler.fit_transform(train["rank"].values.reshape(-1,1))
# train["mat_distance_tort"] = scaler.fit_transform(train["mat_distance_tort"].values.reshape(-1,1))
# train["rank"]= train["rank"]/30
print(train.head)


x_train, y_train = read_tort_train_data(ALL_PATH,h,w)
x_val, y_val = read_tort_val_data(ALL_PATH,h,w)
x_test, y_test = read_tort_test_data(ALL_PATH,h,w)

np.save('x_all_train'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', x_train)
np.save('y_all_train'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', y_train)
np.save('x_all_test'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', x_test)
np.save('y_all_test'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', y_test)
np.save('x_all_val'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', x_val)
np.save('y_all_val'+str( h)+'_'+str(w)+'_v2_nono_dt.npy', y_val)
