import numpy as np
from model import lenet
from keras.utils import np_utils
import keras
from sklearn.model_selection import train_test_split

# input image dimensions
img_rows, img_cols = 28, 28
classes = 10
batch_size = 128
epochs = 12

data = np.loadtxt('data/train.csv', delimiter=',', dtype=np.float32, skiprows=1)
X = data[:, 1:] / 255
y = data[:, 0:1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


# initialize the optimizer and model
model = lenet.LeNet.build((img_rows, img_cols, 1), classes)

model.compile(loss=keras.losses.categorical_crossentropy,
			  optimizer=keras.optimizers.Adadelta(),
			  metrics=['accuracy'])

model.fit(X_train, y_train,
		  batch_size=batch_size,
		  epochs=epochs,
		  verbose=1,
		  validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
model.save_weights('model/weights/lenet_weights.hdf5', overwrite=True)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
