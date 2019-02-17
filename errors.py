import numpy as np
from model.lenet import LeNetV7

seed = 7
np.random.seed(seed)

# input image dimensions
img_rows, img_cols = 28, 28
classes = 10
batch_size = 200
epochs = 10
test_size = 0.1
augm = False
weights_path = 'model/weights/lenetv7_weights_adam30000_256_70_aug.hdf5'


# initialize the optimizer and model
lenet_model = LeNetV7((img_rows, img_cols, 1), classes, weights_path)
(x_train, y_train, x_test, y_test) = lenet_model.load_datasets(test_size)
# Data augmentation
if (augm):
    (x_train, y_train, x_test, y_test) = lenet_model.data_augmentation()

# Load weights
lenet_model.model.load_weights(weights_path)


# Fit the dataset
# lenet_model.fit(x_train, y_train, x_test, y_test, batch_size, epochs)

# Evalutate the model
score = lenet_model.model.evaluate(x_test, y_test, verbose=0)

# Print the score
lenet_model.print_score(score)

# Save weights
# lenet_model.model.save_weights(weights_path)

# Predict and extract errors on a large set of data
errors = lenet_model.get_errors(x_train, y_train, 'Train set')
lenet_model.plot(errors)

# errors = lenet_model.get_errors( X_test, y_test, 'Test set')
# lenet_model.plot(errors)