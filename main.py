import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from load_data import *
import config
C = config.Config()

epochs = C.epochs
input_shape = C.input_shape
batch_size = C.batch_size

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

if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]).astype('float32')
else:
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1).astype('float32')

 # convert class vectors to binary class matrices
num_classes = len(class_names)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
#
# #Model definition
model = Sequential()

model.add(Convolution2D(6, kernel_size=(3, 3), activation='elu', input_shape=input_shape, padding="same"))
model.add(Convolution2D(32, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, kernel_size=(3, 3), border_mode='same', activation='elu'))
model.add(Convolution2D(64, kernel_size=(3, 3), activation='elu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print(model.summary())

#For saving weight
save_filename = 'weights.h5'
callback_period = 5
verbose = 1
if not os.path.exists(save_filename):
    # Model checkpoint callback
    checkpoint = ModelCheckpoint(
        save_filename,
        monitor='val_acc',
        verbose=verbose,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=callback_period)

    # Fit the model
    history = model.fit(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint])

    # List all data in history
    print(history.history.keys())

    # Plot history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('img/acc.png')
    plt.clf()

    # Plot history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('img/loss.png')
else:
    # Load previously saved weights and evaluate the model
    model.load_weights(save_filename)
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score)
    print('Test accuracy:', acc)
