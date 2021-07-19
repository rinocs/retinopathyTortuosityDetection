from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

from utils.models import dataprepare

def createModelOne():
    model = Sequential()
    # first set of CONV => RELU => MAX POOL layers
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=NUM_CLASSES, activation='softmax'))
    # returns our fully constructed deep learning + Keras image classifier 
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    # use binary_crossentropy if there are two classes
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def res_block(X, filter, stage):
      
    # Convolutional_block
    X_copy = X

    f1 , f2, f3 = filter
        
    # Main Path
    X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = MaxPool2D((2,2))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


    # Short path
    X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
    X_copy = MaxPool2D((2,2))(X_copy)
    X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

    # ADD
    X = Add()([X,X_copy])
    X = Activation('relu')(X)

    # Identity Block 1
    X_copy = X


    # Main Path
    X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

    # ADD
    X = Add()([X,X_copy])
    X = Activation('relu')(X)

    # Identity Block 2
    X_copy = X


    # Main Path
    X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
    X = Activation('relu')(X) 

    X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

    # ADD
    X = Add()([X,X_copy])
    X = Activation('relu')(X)

    return X

def createModelTwo():
    input_shape = (256,256,3)

    #Input tensor shape
    X_input = Input(input_shape)

    #Zero-padding

    X = ZeroPadding2D((3,3))(X_input)

    # 1 - stage

    X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides= (2,2))(X)

    # 2- stage

    X = res_block(X, filter= [64,64,256], stage= 2)

    # 3- stage

    X = res_block(X, filter= [128,128,512], stage= 3)

    # 4- stage

    X = res_block(X, filter= [256,256,1024], stage= 4)

    # # 5- stage

    # X = res_block(X, filter= [512,512,2048], stage= 5)

    #Average Pooling

    X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)

    #Final layer

    X = Flatten()(X)
    X = Dense(5, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)


    model = Model( inputs= X_input, outputs = X, name = 'Resnet18')

    model.summary()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics= ['accuracy'])

    return model

# def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
#     # initialize the input shape and channel dimension, assuming
# 	## TensorFlow/channels-last ordering
#     inputShape = (height, width, depth)
#     chanDim = -1
#     # define the model input
#     inputs = Input(shape=inputShape)
# 	# loop over the number of filters
# 	for (i, f) in enumerate(filters):
#         # if this is the first CONV layer then set the input
#         # appropriately
#         if i== 0:
#             x = inputs
#         # CONV => RELU => BN => POOL
#         x = Conv2D(f, (3, 3), padding="same")(x)
#         x = Activation("relu")(x)
#         x = BatchNormalization(axis=chanDim)(x)
#         x = MaxPooling2D(pool_size=(2, 2))(x)
#     # flatten the volume, then FC => RELU => BN => DROPOUT
# 	x = Flatten()(x)
# 	x = Dense(16)(x)
# 	x = Activation("relu")(x)
# 	x = BatchNormalization(axis=chanDim)(x)
# 	x = Dropout(0.5)(x)
# 	# apply another FC layer, this one to match the number of nodes
# 	# coming out of the MLP
# 	x = Dense(4)(x)
# 	x = Activation("relu")(x)
# 	# check to see if the regression node should be added
# 	if regress:
# 		x = Dense(1, activation="linear")(x)
# 	# construct the CNN
# 	model = Model(inputs, x)
# 	# return the CNN
# 	return model



def create_model_reg(image_size):
    nb_filters = 8
    nb_conv = 5

    model = Sequential()
    model.add(Conv2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(image_size, image_size,1) ) )
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('linear'))

    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model


def create_model_reg_2(image_size):
    model = Sequential()

    # model.add(Conv2D(32, (3, 3), strides = (1, 1), name = 'conv0', input_shape = (1,image_size, image_size)))
    model.add(Conv2D(64, (3, 3),strides = (1, 1), name = 'conv0',padding='same', input_shape=(64,64,1)))

    # model.add(BatchNormalization(axis = 3, name = 'bn0'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), name='max_pool'))
    # model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
    model.add(Conv2D(64, (3, 3), name="conv1"))
    model.add(Activation('relu'))
    # model.add(AveragePooling2D((3, 3), name='avg_pool'))
    model.add(MaxPooling2D((3, 3), name='max1_pool'))

    model.add(GlobalAveragePooling2D())
    model.add(Dense(300, activation="relu", name='rl'))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation='relu', name='sm'))
    print(model.summary())
    # model.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-5), metrics=['accuracy'])
    model.compile(loss='mean_squared_error',optimizer=Adam(lr=1e-5),metrics=['mean_squared_error'])
    return model

def create_cnn_custom(width, height, depth, filters=(16, 32, 64), regress=False):
    inputShape = (height,width,depth)
    chanDim = -1
    
    #define the input 
    inputs = Input(shape=inputShape)
    
    # loop over filters 
    for (i,f) in enumerate(filters):
        # first layer
        if i==0: 
            x=inputs
        # CONV => RELU => BN =>POOL
        x = Conv2D(f, (3,3), padding='same')(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2,2))(x)
        # x = Dropout(0.25)(x)
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)
    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)
    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)
    # construct the CNN
    model = Model(inputs, x)
        # return the CNN
    return model
    
    
def train_model(image_size,batch_size = 50, nb_epoch = 20 ):
    num_samples = 1999
    cv_size = 499

    train_data, train_target = dataprepare.read_and_normalize_train_data()
    train_data = train_data[0:num_samples,:,:,:]
    train_target = train_target[0:num_samples]

    X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_target, test_size=cv_size, random_state=56741)

    model = create_model_reg_2()
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_valid, y_valid) )

    predictions_valid = model.predict(X_valid, batch_size=50, verbose=1)
    compare = pd.DataFrame(data={'original':y_valid.reshape((cv_size,)),'prediction':predictions_valid.reshape((cv_size,))})
    compare.to_csv('compare.csv')

    return model, history
