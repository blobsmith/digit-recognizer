from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from model.abstract import Abstract
from keras import losses, optimizers, initializers
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
        model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(15, (3, 3), activation='relu'))
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