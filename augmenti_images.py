import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img




rot_path="sample/tort/skeletons"
save_path ="sample/tort/skeletons/aug_images"


image_w, image_h = 427,53
batch_size = 32

x_train= np.load('./x_train_53_427tort_rank.npy') 
y_train=np.load('./y_train_53_427tort_rank.npy')

datagen = ImageDataGenerator(
        # rescale=1./255,
        # rotation_range=3,
        # shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        
       )
x_train = x_train.reshape(-1, image_h, image_w,1)
train_iterator = datagen.flow(x_train, y_train, batch_size=batch_size,save_to_dir=save_path,save_prefix='aug', save_format='png')
it = datagen.flow(x_train,y_train, batch_size=batch_size)
# # generate samples and plot
for i in range(50):
    # define subplot
   
    # generate batch of images
    batch = train_iterator.next()
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # image = image[:,:,0]
    # plot raw pixel data
    plt.imshow(np.squeeze(image[0]), cmap="gray")
    plt.show()
# show the figure