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

submission_lenetv2_weights_adadelta0.1.csv
---------------------------------------------------------------------------------
LeNetV2 - 40 Epochs - 37800 training data - Optimizer Adadelta
ReduceLR
test_size=0.1

Kaggle: 0.98814

submission_lenetv2_weights_adadelta0.1_reducelr.csv
----------------------------------------------------------------------------------
LeNetV3 - 15 Epochs - 600000 training data - Opimizer Adam 
10000 test data (using the real minst dataset, not the one provided by kaggle)

loss: 0.0111 - acc: 0.9962 - val_loss: 0.0203 - val_acc: 0.9930
loss: 0.0203088142807
acc: 99.3%

Kaggle: 0.99885 (top 85, You advanced 1,041 places on the leaderboard!)

submission_lenetv3_weights_adam10000_256_15.csv
------------------------------------------------------------------------------------
LeNetV4 - 30 Epochs - 600000 training data - Opimizer Adam 
10000 test data (using the real minst dataset, not the one provided by kaggle)

loss: 0.0085 - acc: 0.9974 - val_loss: 0.0333 - val_acc: 0.9924
loss: 0.0332907435369
acc: 99.24%
------------------------------------------------------------------------------------
LeNetV4 - 27 Epochs - 600000 training data - Opimizer Adam - add 1 dropout -> 0.2
10000 test data (using the real minst dataset, not the one provided by kaggle)

loss: 0.0093 - acc: 0.9970 - val_loss: 0.0329 - val_acc: 0.9927
loss: 0.0329076314679
acc: 99.27%
------------------------------------------------------------------------------------
LeNetV4 - 27 Epochs - 600000 training data - Opimizer Adam - add 3 dropout -> 0.2
10000 test data (using the real minst dataset, not the one provided by kaggle)

loss: 0.0127 - acc: 0.9963 - val_loss: 0.0367 - val_acc: 0.9926
loss: 0.0367099021842
acc: 99.26%
best: (loss: 0.0242 - acc: 0.9930 - val_loss: 0.0329 - val_acc: 0.9927)

------------------------------------------------------------------------------------
LeNetV4 - 60 Epochs - 600000 training data - Opimizer Adam - add 3 dropout all dropout from 0.2 to -> 0.5
10000 test data (using the real minst dataset, not the one provided by kaggle)

loss: 0.0319 - acc: 0.9929 - val_loss: 0.0454 - val_acc: 0.9930
loss: 0.0453695610985
acc: 99.3%