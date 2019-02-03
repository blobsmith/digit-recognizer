import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import math

class LeNet:

    @staticmethod
    def build(input_shape = (28, 28, 1), classes = 10):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))

        return model

    @staticmethod
    def predict(model, X_new, y_new, context):
        errors = []
        y_at = model.predict_classes(X_new)
        # show the inputs and predicted outputs
        for i in range(len(X_new)):
            if y_at[i] != np.argmax(y_new[i]):
                errors.append({
                    'x': X_new[i],
                    'y': np.argmax(y_new[i]),
                    'prediction': y_at[i],
                    'context': context
                })
        return errors


    @staticmethod
    def plot(errors):
        fig = plt.figure(figsize=(50, 50))
        columns = 4
        rows = 4
        index = 1
        for error in errors:
            fig.add_subplot(rows, columns, index)
            plt.imshow(1 - error['x'][:, :, 0], cmap='gray')
            plt.text(2, 4, str(error['prediction']), fontsize=30, bbox=dict(boxstyle="round", facecolor='red', alpha=0.5))
            plt.text(24, 4, str(error['y']), fontsize=30, bbox=dict(boxstyle="round", facecolor='green', alpha=0.5))
            plt.title(error['context'])

            if index % (rows*columns) == 0:
                plt.show()
                index = 1
                fig = plt.figure(figsize=(50, 50))
            else:
                index = index + 1

        plt.show()