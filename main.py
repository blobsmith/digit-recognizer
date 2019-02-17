import numpy as np
from model.lenet import LeNetV7

# input image dimensions
img_rows, img_cols = 28, 28
classes = 10
batch_size = 256
epochs = 70
test_size = 30000
augm = True
weights_path = 'model/weights/lenetv7_weights_adam'+str(test_size)+'_'+str(batch_size)+'_'+str(epochs)+'_'+('aug' if augm else '')+'.hdf5'

# initialize the optimizer and model
lenet_model = LeNetV7((img_rows, img_cols, 1), classes, weights_path)

# Load data
(x_train, y_train, x_test, y_test) = lenet_model.load_datasets(test_size)

# Data augmentation
(x_train_aug, y_train_aug, x_test_aug, y_test_aug) = lenet_model.data_augmentation()

# Fit the dataset
lenet_model.fit(x_train_aug, y_train_aug, x_test, y_test, batch_size, epochs)

# Evalutate the model
score = lenet_model.model.evaluate(x_test, y_test, verbose=0)

# Print the score
lenet_model.print_score(score)
