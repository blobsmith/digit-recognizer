import numpy as np
from model import lenet
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


# input image dimensions
img_rows, img_cols = 28, 28
classes = 10
batch_size = 128
epochs = 12
weights_path = 'model/weights/lenet_weights.hdf5'

# load data from csv
data = np.loadtxt('data/train.csv', delimiter=',', dtype=np.float32, skiprows=1)

# normalize gray scale value (0 -> 255)  to (0 -> 1)
X = data[:, 1:] / 255
y = data[:, 0:1].astype(int)

# set only 0.04% of training set for the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.04)

# reshape from (m, 784) to (m, 28, 28, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# convert labels to one hot vector
y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)


# initialize the optimizer and model
model = lenet.LeNet.build((img_rows, img_cols, 1), classes)

# model.compile(loss=keras.losses.categorical_crossentropy,
# 			  optimizer=keras.optimizers.Adadelta(),
# 			  metrics=['accuracy'])
#
# model.fit(X_train, y_train,
# 		  batch_size=batch_size,
# 		  epochs=epochs,
# 		  verbose=1,
# 		  validation_data=(X_test, y_test))
# score = model.evaluate(X_test, y_test, verbose=0)
model.load_weights(weights_path)


# errors = lenet.LeNet.predict(model, X_train, y_train, 'Train set')
# lenet.LeNet.plot(errors)

# errors = lenet.LeNet.predict(model, X_test, y_test, 'Test set')
# lenet.LeNet.plot(errors)