import os
import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
# from keras.callbacks import ModelCheckpoint
from load_data import *

# epochs = 20
# input_shape = (28, 28, 1)
# batch_size = 128

x_train, y_train, x_test, y_test, class_names = load_data('data')

# Reshape and normalize
x_train = x_train.reshape(x_train.shape[0], 28, 28).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28).astype('float32')

x_train /= 255.0
x_test /= 255.0

print('Training data: ', x_train.shape)
print('Training labels: ', y_train.shape)
print('Test data: ', x_test.shape)
print('Test labels: ', y_test.shape)
print('Class names:', class_names)

plt.figure(figsize=(10, 10))
visualize(x_train, y_train, class_names)

# # convert class vectors to binary class matrices
# num_classes = len(class_names)
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# #Model definition
# model = Sequential()
#
# model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding="same"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(16, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(120, (3, 3), activation='relu'))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(84, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])
#
# print(model.summary())
#
# #For saving weight
# save_filename = 'weights.h5'
# callback_period = 5
# verbose = 1
# if not os.path.exists(save_filename):
#     # Model checkpoint callback
#     checkpoint = ModelCheckpoint(
#         save_filename,
#         monitor='val_acc',
#         verbose=verbose,
#         save_best_only=True,
#         save_weights_only=False,
#         mode='auto',
#         period=callback_period)
#
#     # Fit the model
#     history = model.fit(
#         x_train, y_train,
#         epochs=epochs, batch_size=batch_size, verbose=verbose,
#         validation_data=(x_test, y_test),
#         callbacks=[checkpoint])
#
#     # List all data in history
#     print(history.history.keys())
#
#     # Plot history for accuracy
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('img/acc.png')
#     plt.clf()
#
#     # Plot history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.savefig('img/loss.png')
# else:
#     # Load previously saved weights and evaluate the model
#     model.load_weights(save_filename)
#     score, acc = model.evaluate(x_test, y_test, verbose=0)
#     print('Test score:', score)
#     print('Test accuracy:', acc)
