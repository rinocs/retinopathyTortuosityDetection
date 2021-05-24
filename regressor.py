import numpy as np
import os
import cv2
import pandas as pd
import h5py

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint

from utils.models import dataprepare
from utils.models import models

#### Path 

trainSplittedPath ='sample/CHASE/train/label/splitted_vessels/'
testSplittedPath ='sample/CHASE/test/label/splitted_vessels/'
validateSplittedPath = 'sample/CHASE/validate/label/splitted_vessels/'

### parameters
image_size = 64
batch_size = 32
nb_epoch = 50


### load data

x_train, y_train = dataprepare.read_and_normalize_train_data(trainSplittedPath, image_size)
x_test, y_test = dataprepare.read_and_normalize_test_data(testSplittedPath, image_size)
x_val, y_val = dataprepare.read_and_normalize_val_data(validateSplittedPath, image_size)
x_train = x_train.reshape(-1, 64, 64, 1)
x_val = x_val.reshape(-1, 64, 64, 1)
print("succesful loaded ")


### train model
# Train model
model = models.create_model_reg_2(image_size)
#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="checkpoint/weights.hdf5", verbose=1, save_best_only=True)
history = model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data= (x_val, y_val),
              epochs=nb_epoch,
              shuffle=True,
              callbacks=[checkpointer , earlystopping],
              verbose=2)



# model,history = models.train_model(nb_epoch)