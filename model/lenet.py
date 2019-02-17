from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from model.abstract import Abstract
from keras import losses, optimizers, initializers
from keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from imgaug import augmenters as iaa

seed = 7
np.random.seed(seed)

import keras

class LeNet(Abstract):

    def __init__(self, input_shape = (28, 28, 1), classes = 10):
        super().__init__(input_shape, classes)

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes, activation='softmax'))
        self.model = model

    def compile(self):
        self.model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.Adadelta(),
                      metrics=['accuracy'])


    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs):
        callbacks = [
            keras.callbacks.TensorBoard(
                log_dir='model/logs',
                histogram_freq=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.02,
                patience=2
            )
        ]

        self.model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(X_test, y_test))#,
                  #callbacks=callbacks)

    def load_datasets(self, test_size):
        # load data from csv
        data = np.loadtxt('data/train.csv', delimiter=',', dtype=np.float32, skiprows=1)

        # normalize gray scale value (0 -> 255)  to (0 -> 1)
        X = data[:, 1:] / 255
        y = data[:, 0:1].astype(int)

        # set only 0.04% of training set for the test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # reshape from (m, 784) to (m, 28, 28, 1)
        x_train = x_train.reshape(x_train.shape[0], self.input_shape[0], self.input_shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], self.input_shape[0], self.input_shape[1], 1)

        # convert labels to one hot vector
        y_train = np_utils.to_categorical(y_train, self.classes)
        y_test = np_utils.to_categorical(y_test, self.classes)
        return (x_train, y_train, x_test, y_test)

class LeNetV2(LeNet):

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes, activation='softmax'))
        self.model = model

    def compile(self):
        self.model.compile(loss=losses.categorical_crossentropy,
                      optimizer=optimizers.adadelta(),
                      metrics=['accuracy'])

class LeNetV3(LeNet):

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))
        self.model = model

    def compile(self):
        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.adam(),
                           metrics=['accuracy'])

    def load_datasets(self, test_size):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        return (X_train, y_train, X_test, y_test)

class LeNetV4(LeNetV3):

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(self.classes, activation='softmax'))
        self.model = model

    def compile(self):
        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.adam(lr = 0.0007),
                           metrics=['accuracy'])

    def data_augmentation(self, ):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        seq = iaa.Sequential([
            iaa.Add((-150, -45), per_channel=True),
            iaa.GammaContrast(gamma=1.44)
        ])
        x_train_aug = seq.augment_images(X_train)
        x_test_aug = seq.augment_images(X_test)

        X_train = np.concatenate((X_train, x_train_aug), axis=0)
        X_test = np.concatenate((X_test, x_test_aug), axis=0)
        y_train = np.concatenate((y_train, y_train), axis=0)
        y_test = np.concatenate((y_test, y_test), axis=0)

        return(X_train/255, y_train, X_test/255, y_test)

class LeNetV5(LeNetV4):

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.classes, activation='softmax'))
        self.model = model

    def compile(self):
        self.model.compile(loss=losses.categorical_crossentropy,
                           optimizer=optimizers.adam(),
                           metrics=['accuracy'])


    def data_augmentation(self, ):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        # Add dark images
        seq = iaa.Sequential([
            iaa.Add((-150, -45), per_channel=True),
            iaa.GammaContrast(gamma=1.44)
        ])
        x_train_aug = seq.augment_images(X_train)
        x_test_aug = seq.augment_images(X_test)

        X_train = np.concatenate((X_train, x_train_aug), axis=0)
        X_test = np.concatenate((X_test, x_test_aug), axis=0)
        y_train = np.concatenate((y_train, y_train), axis=0)
        y_test = np.concatenate((y_test, y_test), axis=0)

        # Add blur images
        seq = iaa.Sequential([
            iaa.MotionBlur(k=9, angle=0)
        ])

        # show an image with 8*8 augmented versions of image 0
        # seq.show_grid([X_train[0], X_train[1], X_train[2]], cols=2, rows=2)
        x_train_aug = seq.augment_images(X_train)
        x_test_aug = seq.augment_images(X_test)

        X_train = np.concatenate((X_train, x_train_aug), axis=0)
        X_test = np.concatenate((X_test, x_test_aug), axis=0)
        y_train = np.concatenate((y_train, y_train), axis=0)
        y_test = np.concatenate((y_test, y_test), axis=0)

        return (X_train / 255, y_train, X_test / 255, y_test)

class LeNetV6(LeNetV5):

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(self.classes, activation='softmax'))
        self.model = model




class LeNetV7(LeNetV6):

    def __init__(self, input_shape = (28, 28, 1), classes = 10, weights_path = ''):
        super().__init__(input_shape, classes)
        self.weights_path = weights_path

    def build(self):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(self.classes, activation='softmax'))
        self.model = model


    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs):
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=self.weights_path,
                monitor='val_loss',
                save_best_only=True
            ),
        ]

        self.model.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,
                  validation_data=(X_test, y_test),
                  callbacks=callbacks)