import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

class Abstract(ABC):

    def __init__(self, input_shape, classes):
        self.input_shape = input_shape
        self.classes = classes
        self.build()
        self.compile()

    @abstractmethod
    def build(self):
        self.model = None
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def fit(self, X_train, y_train, X_test, y_test, batch_size, epochs):
        pass

    def get_errors(self, X_new, y_new, context):
        errors = []
        y_at = self.model.predict_classes(X_new)
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

    def print_score(self, score):
        index = 0
        for label in self.model.metrics_names:
            if label == 'acc':
                print(label + ': ' + str(score[index] * 100) + '%')
            else:
                print(label + ': ' + str(score[index]))
            index = index + 1

    @staticmethod
    def plot(errors):
        fig = plt.figure(figsize=(50, 50))
        columns = 4
        rows = 4
        index = 1
        for error in errors:
            fig.add_subplot(rows, columns, index)
            plt.imshow(1 - error['x'][:, :, 0], cmap='gray')
            plt.text(2, 4, str(error['prediction']), fontsize=30,
                     bbox=dict(boxstyle="round", facecolor='red', alpha=0.5))
            plt.text(24, 4, str(error['y']), fontsize=30, bbox=dict(boxstyle="round", facecolor='green', alpha=0.5))
            plt.title(error['context'])

            if index % (rows * columns) == 0:
                plt.show()
                index = 1
                fig = plt.figure(figsize=(50, 50))
            else:
                index = index + 1

        plt.show()