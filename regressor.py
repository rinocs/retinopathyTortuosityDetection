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
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
from utils.models import dataprepare, models



logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def print_evaluate(true, predicted):  
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)

### parameters
image_size = 128
batch_size = 32
nb_epoch = 20



x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
x_val = np.load('x_val.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
y_val = np.load('y_val.npy')

x_train = x_train.reshape(-1, image_size, image_size, 1)
x_val = x_val.reshape(-1, image_size, image_size, 1)



scaler = MinMaxScaler()
# transform data
y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test = scaler.fit_transform(y_test.reshape(-1,1))
y_val = scaler.fit_transform(y_val.reshape(-1,1))
logging.info("succesful loaded")
print("succesful loaded ")


### train model
# Train model
# model = models.create_model_reg_2(image_size)
logging.info('creating model')
print("creating model")
model = models.create_cnn_custom(image_size, image_size, 1, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
# model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mean_squared_error'])
model.compile(loss="mean_absolute_percentage_error", optimizer=opt, metrics=['mean_absolute_percentage_error'])
print(model.summary())
#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)

#save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="checkpoint/weights.hdf5", verbose=1, save_best_only=True)
logging.info("Training Model")
print("training model")
history = model.fit(x_train, y_train,
              batch_size=batch_size,
              validation_data= (x_val, y_val),
              epochs=nb_epoch,
              shuffle=True,
              callbacks=[checkpointer , earlystopping],
              verbose=2)


logging.info("finished Train")
print("training finished")

logging.info("predicting tortuosity")
print("[INFO] predicting tortuosity...")
preds = model.predict(x_test)

print(model.evaluate(x_test, y_test))

# compute the difference between the *predicted* tortuosity and the
# *actual* tortuosity, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - y_test
percentDiff = (diff / y_test) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

print("AbsPercentDiff: ", absPercentDiff)
print('mean: ', mean)
print ("std: ", std)
print("preds: ", preds)
print("y_test: ", y_test)

plt.scatter(y_test, preds)
plt.show()

residuals = y_test - preds
# sns.displot(residuals)

# list all data in history
print(history.history.keys())
# summarize history for loss
plot_loss(history)