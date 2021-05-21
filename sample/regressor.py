import numpy as np
import os
import cv2
import pandas as pd

from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop, Adam, Adadelta

image_size = 50







train_model(nb_epoch = 50)