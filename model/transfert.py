import sys
import numpy as np
import cv2
import sklearn.metrics as sklm

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model

# from keras.datasets import mnist
from sklearn.model_selection import train_test_split

from keras import backend as K

img_dim_ordering = 'tf'
K.set_image_dim_ordering(img_dim_ordering)


# the model
def pretrained_model(img_shape, num_classes):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    # model_vgg16_conv.summary()

    # Create your own input format
    keras_input = Input(shape=img_shape, name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(keras_input)
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Add the fully-connected layers
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create your own model
    pretrained_model = Model(inputs=keras_input, outputs=x)
    pretrained_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return pretrained_model


# loading the data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# load data from csv
data = np.loadtxt('../data/train.csv', delimiter=',', dtype=np.float32, skiprows=1)

# normalize gray scale value (0 -> 255)  to (0 -> 1)
X = data[:, 1:] / 255
y = data[:, 0:1].astype(int)

# set only 0.04% of training set for the test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# converting it to RGB
x_train = [cv2.cvtColor(cv2.resize(i, (32, 32)), cv2.COLOR_GRAY2BGR) for i in x_train]
x_train = np.concatenate([arr[np.newaxis] for arr in x_train]).astype('float32')

x_test = [cv2.cvtColor(cv2.resize(i, (32, 32)), cv2.COLOR_GRAY2BGR) for i in x_test]
x_test = np.concatenate([arr[np.newaxis] for arr in x_test]).astype('float32')

# training the model
model = pretrained_model(x_train.shape[1:], len(y_train))
hist = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), verbose=1)