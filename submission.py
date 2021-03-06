import numpy as np
from model.lenet import LeNetV7


# input image dimensions
img_rows, img_cols = 28, 28
classes = 10
weights_path = 'model/weights/lenetv7_weights_adam30000_256_70_aug.hdf5'

# load data from csv
data = np.loadtxt('data/test.csv', delimiter=',', dtype=np.float32, skiprows=1)

# normalize gray scale value (0 -> 255)  to (0 -> 1)
X = data / 255

# reshape from (m, 784) to (m, 28, 28, 1)
X = X.reshape(X.shape[0], img_rows, img_cols, 1)

# initialize the optimizer and model
model_lenet = LeNetV7((img_rows, img_cols, 1), classes)
model_lenet.model.load_weights(weights_path)
predictions = model_lenet.model.predict_classes(X)


submissions = [['ImageId','Label']]

index = 1
with open('data/lenetv7_weights_adam30000_256_70_aug.csv', 'a') as the_file:
    the_file.write('ImageId,Label\n')
    for y in predictions:
        the_file.write(str(index)+','+str(y)+'\n')
        index = index + 1