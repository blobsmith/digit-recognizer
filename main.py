import numpy as np
from model.lenet import LeNet, LeNetV2, LeNetV3

seed = 7
np.random.seed(seed)

# input image dimensions
img_rows, img_cols = 28, 28
classes = 10
batch_size = 256
epochs = 30
test_size = 10000
weights_path = 'model/weights/lenetv3_weights_adam'+str(test_size)+'_'+str(batch_size)+'_'+str(epochs)+'.hdf5'

# initialize the optimizer and model
lenet_model = LeNetV3((img_rows, img_cols, 1), classes)
(x_train, y_train, x_test, y_test) = lenet_model.load_weights(test_size)

# Fit the dataset
lenet_model.fit(x_train, y_train, x_test, y_test, batch_size, epochs)

# Evalutate the model
score = lenet_model.model.evaluate(x_test, y_test, verbose=0)

# Print the score
lenet_model.print_score(score)

# Save weights
lenet_model.model.save_weights(weights_path)