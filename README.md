# digit-recognizer

Standard model - LeNet
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

12 Epochs - 40320 training data - Optimizer Adadelta

test_size=0.04

102s 3ms/step - loss: 0.0265 - acc: 0.9919 - val_loss: 0.0416 - val_acc: 0.9839

Test loss: 0.041601706746496346
Test accuracy: 0.9839285714285714
Kaggle: 0.98871

----------------------------------------------------------------------------------
LeNet - 12 Epochs - 37800 training data - Optimizer Adadelta

test_size=0.1

16s 416us/step - loss: 0.0265 - acc: 0.9919 - val_loss: 0.0491 - val_acc: 0.9886

Test loss: 0.009815748790653223
Test accuracy: 0.9971428571428571

Kaggle: 0.98985

---------------------------------------------------------------------------------
LeNet - 12 Epochs - 37800 training data - Optimizer Adam

test_size=0.1

loss: 0.0235 - acc: 0.9924 - val_loss: 0.0509 - val_acc: 0.9902
Test loss: 0.05091213624304926
Test accuracy: 0.9902380952380953

---------------------------------------------------------------------------------
LeNet - 20 Epochs - 37800 training data - Optimizer Adam

test_size=0.1

loss: 0.0151 - acc: 0.9945 - val_loss: 0.0304 - val_acc: 0.9926
Test loss: 0.03042094535018361
Test accuracy: 0.9926190476190476
Kaggle: 0.98942

---------------------------------------------------------------------------------
LeNetV2 - 40 Epochs - 37800 training data - Optimizer Adadelta

test_size=0.1

Kaggle: 0.98942

submission_lenetv2_weights_adadelta0.1.hdf5
---------------------------------------------------------------------------------
LeNetV2 - 40 Epochs - 37800 training data - Optimizer Adadelta
ReduceLR
test_size=0.1

Kaggle: 0.98814

submission_lenetv2_weights_adadelta0.1_reducelr.hdf5