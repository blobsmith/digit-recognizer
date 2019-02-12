import numpy as np
from model.lenet import LeNet, LeNetV2, LeNetV3
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

seed = 7
np.random.seed(seed)

# input image dimensions
img_rows, img_cols = 28, 28
classes = 10
batch_size = 200
epochs = 10
test_size = 0.1
weights_path = 'model/weights/lenetv3_weights_adam'+str(test_size)+'_'+str(batch_size)+'_'+str(epochs)+'.hdf5'

# load data from csv
data = np.loadtxt('data/train.csv', delimiter=',', dtype=np.float32, skiprows=1)

# normalize gray scale value (0 -> 255)  to (0 -> 1)
X = data[:, 1:] / 255
y = data[:, 0:1].astype(int)

# set only 0.04% of training set for the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# reshape from (m, 784) to (m, 28, 28, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

# convert labels to one hot vector
y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)


# initialize the optimizer and model
lenet_model = LeNetV3((img_rows, img_cols, 1), classes)

# Load weights
# lenet_model.model.load_weights(weights_path)


# Fit the dataset
lenet_model.fit(X_train, y_train, X_test, y_test, batch_size, epochs)

# Evalutate the model
score = lenet_model.model.evaluate(X_test, y_test, verbose=0)

# Print the score
lenet_model.print_score(score)

# Save weights
lenet_model.model.save_weights(weights_path)

# Predict and extract errors on a large set of data
# errors = lenet_model.predict(model, X_train, y_train, 'Train set')
# lenet_model.plot(errors)

# errors = lenet_model.predict(model, X_test, y_test, 'Test set')
# lenet_model.plot(errors)