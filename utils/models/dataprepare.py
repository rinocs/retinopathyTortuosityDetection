
import pandas as pd
import os
import cv2
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def load_train(train_path, h,w):
    X_train = []
    y_train = []
    tort = pd.read_csv(train_path+'train_splitted.csv')
    print('Read train images')
    for index, row in tort.iterrows():
        image_path = os.path.join(train_path, str(row['image']) )
        print(image_path)
        img = cv2.imread(image_path, 0)
        # print(img.shape)
        img = cv2.resize(img, (w, h) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        # img = img/255
        X_train.append(img)
        # y_train.append( [ row['VTI'] ] )
        y_train.append( [ row['distance_tort'] ] )
    return X_train, y_train

def load_test(test_path, h,w):
    X_test = []
    y_test = []
    tort = pd.read_csv(test_path+'test_splitted.csv')
    print('Read test images')
    for index, row in tort.iterrows():
        image_path = os.path.join(test_path, str(row['image']))
        img = cv2.resize(cv2.imread(image_path, 0), (w, h) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        # img = img/255
        X_test.append(img)
        # y_test.append( [ row['VTI'] ] )
        y_test.append( [ row['distance_tort'] ] )
    return X_test, y_test

def load_val(val_path, h,w):
    X_val = []
    y_val = []
    tort = pd.read_csv(val_path+'validate_splitted.csv')
    print('Read val images')
    for index, row in tort.iterrows():
        image_path = os.path.join(val_path, str(row['image']))
        img = cv2.resize(cv2.imread(image_path, 0), (w, h) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        # img = img/255
        X_val.append(img)
        # y_val.append( [ row['VTI'] ] )
        y_val.append( [ row['distance_tort'] ] )
    return X_val, y_val
def load_tort(tort_path, h,w):
    X_val = []
    y_val = []
    tort = pd.read_csv(tort_path+'all_tort.csv')
    print('Read tort images')
    for index, row in tort.iterrows():
        image_path = os.path.join(tort_path, str(row['image']))
        img = cv2.resize(cv2.imread(image_path, 0), (w, h) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        # img = img/255
        X_val.append(img)
        # y_val.append( [ row['VTI'] ] )
        y_val.append( [ row['rank'] ] )
    return X_val, y_val
def load_tort_dt(val_path, h,w):
    X_val = []
    y_val = []
    tort = pd.read_csv(val_path+'all_tort.csv')
    print('Read tort images')
    for index, row in tort.iterrows():
        image_path = os.path.join(val_path, str(row['image']))
        img = cv2.resize(cv2.imread(image_path, 0), (w, h) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        # img = img/255
        X_val.append(img)
        # y_val.append( [ row['VTI'] ] )
        y_val.append( [ row['distance_tort'] ] )
    return X_val, y_val

def read_and_normalize_train_data(path, h,w):
    train_data, train_target = load_train(path, h,w)
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    m = train_data.mean()
    s = train_data.std()

    print ('Train mean, sd:', m, s )
    train_data -= m
    train_data /= s
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

def read_and_normalize_test_data(path, h,w):
    test_data, test_target = load_test(path,  h,w)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.float32)
    m = test_data.mean()
    s = test_data.std()

    print ('test mean, sd:', m, s )
    test_data -= m
    test_data /= s
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_target

def read_and_normalize_val_data(path,  h,w):
    val_data, val_target = load_val(path,  h,w)
    val_data = np.array(val_data, dtype=np.float32)
    val_target = np.array(val_target, dtype=np.float32)
    m = val_data.mean()
    s = val_data.std()

    print ('Validation mean, sd:', m, s )
    val_data -= m
    val_data /= s
    print('Validate shape:', val_data.shape)
    print(val_data.shape[0], 'validatation samples')
    return val_data, val_target

def read_and_normalize_train_data_v1(path,  h,w):
    train_data, train_target = load_train(path,  h,w)
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    train_data = scaler.fit_transform(train_data.reshape(-1,1))
    
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

def read_and_normalize_test_data_v1(path,  h,w):
    test_data, test_target = load_test(path,  h,w)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.float32)
    test_data = scaler.fit_transform(test_data.reshape(-1,1))
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_target

def read_and_normalize_val_data_v1(path,  h,w):
    val_data, val_target = load_val(path,  h,w)
    val_data = np.array(val_data, dtype=np.float32)
    val_target = np.array(val_target, dtype=np.float32)
    val_data = scaler.fit_transform(val_data.reshape(-1,1))
    print('Validate shape:', val_data.shape)
    print(val_data.shape[0], 'validatation samples')
    return val_data, val_target

def read_and_normalize_train_data_v2(path,  h,w):
    train_data, train_target = load_train(path,  h,w)
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    train_data = train_data/255
    
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

def read_and_normalize_test_data_v2(path,  h,w):
    test_data, test_target = load_test(path,  h,w)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.float32)
    test_data = test_data/255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_target

def read_and_normalize_val_data_v2(path,  h,w):
    val_data, val_target = load_val(path,  h,w)
    val_data = np.array(val_data, dtype=np.float32)
    val_target = np.array(val_target, dtype=np.float32)
    val_data = val_data/255
    print('Validate shape:', val_data.shape)
    print(val_data.shape[0], 'validatation samples')
    return val_data, val_target

def read_train_data_v2(path,  h,w):
    train_data, train_target = load_train(path,  h,w)
    train_data = np.array(train_data, dtype=np.float32)
    train_target = np.array(train_target, dtype=np.float32)
    # train_data = train_data/255
    
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

def read_test_data_v2(path,  h,w):
    test_data, test_target = load_test(path,  h,w)
    test_data = np.array(test_data, dtype=np.float32)
    test_target = np.array(test_target, dtype=np.float32)
    # test_data = test_data/255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_target

def read_val_data_v2(path,  h,w):
    val_data, val_target = load_val(path,  h,w)
    val_data = np.array(val_data, dtype=np.float32)
    val_target = np.array(val_target, dtype=np.float32)
    # val_data = val_data/255
    print('Validate shape:', val_data.shape)
    print(val_data.shape[0], 'validatation samples')
    return val_data, val_target

def read_tort_data(path,  h,w):
    tort_data, tort_target = load_tort(path,  h,w)
    tort_data = np.array(tort_data, dtype=np.float32)
    tort_target = np.array(tort_target, dtype=np.float32)
    # val_data = val_data/255
    print('tort shape:', tort_data.shape)
    print(tort_data.shape[0], 'tort samples')
    return tort_data, tort_target

def read_tort_data_dt(path,  h,w):
    tort_data, tort_target = load_tort_dt(path,  h,w)
    tort_data = np.array(tort_data, dtype=np.float32)
    tort_target = np.array(tort_target, dtype=np.float32)
    # val_data = val_data/255
    print('tort shape:', tort_data.shape)
    print(tort_data.shape[0], 'tort samples')
    return tort_data, tort_target