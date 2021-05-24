
import pandas as pd
import os
import cv2
import numpy as np
import math


def load_train(train_path, image_size):
    X_train = []
    y_train = []
    tort = pd.read_csv(train_path+'train_splitted.csv')
    print('Read train images')
    for index, row in tort.iterrows():
        image_path = os.path.join(train_path, str(row['image']) + '.png')
        img = cv2.resize(cv2.imread(image_path, 0), (image_size, image_size) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        X_train.append(img)
        y_train.append( [ row['VTI'] ] )
    return X_train, y_train

def load_test(test_path, image_size):
    X_test = []
    y_test = []
    tort = pd.read_csv(test_path+'test_splitted.csv')
    print('Read test images')
    for index, row in tort.iterrows():
        image_path = os.path.join(test_path, str(row['image']) + '.png')
        img = cv2.resize(cv2.imread(image_path, 0), (image_size, image_size) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        X_test.append(img)
        y_test.append( [ row['VTI'] ] )
    return X_test, y_test

def load_val(val_path, image_size):
    X_val = []
    y_val = []
    tort = pd.read_csv(val_path+'validate_splitted.csv')
    print('Read val images')
    for index, row in tort.iterrows():
        image_path = os.path.join(val_path, str(row['image']) + '.png')
        img = cv2.resize(cv2.imread(image_path, 0), (image_size, image_size) ).astype(np.float32)
        # img = img.transpose((2,0,1))
        X_val.append(img)
        y_val.append( [ row['VTI'] ] )
    return X_val, y_val

def read_and_normalize_train_data(path, image_size):
    train_data, train_target = load_train(path, image_size)
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

def read_and_normalize_test_data(path, image_size):
    test_data, test_target = load_test(path, image_size)
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

def read_and_normalize_val_data(path, image_size):
    val_data, val_target = load_val(path, image_size)
    val_data = np.array(val_data, dtype=np.float32)
    val_target = np.array(val_target, dtype=np.float32)
    m = val_data.mean()
    s = val_data.std()

    print ('Validation mean, sd:', m, s )
    val_data -= m
    val_data /= s
    print('Validate shape:', val_data.shape)
    print(val_data.shape[0], 'train samples')
    return val_data, val_target
