import os
import numpy as np
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from load_data import *
from model import load_model
import config
C = config.Config()

epochs      = C.epochs
batch_size  = C.batch_size
image_size  = C.image_size

x_train, y_train, x_test, y_test, class_names = load_data('data')
num_classes = len(class_names)

# Reshape and normalize
x_train = x_train.reshape(x_train.shape[0], image_size, image_size).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_size, image_size).astype('float32')

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
    x_train = x_train.reshape(x_train.shape[0], 1, image_size, image_size).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, image_size, image_size).astype('float32')
    input_shape = (1, image_size, image_size)
else:
    x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')
    input_shape = (image_size, image_size, 1)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Load model
model = load_model(input_shape, num_classes)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print(model.summary())

save_filename   = C.save_filename
callback_period = C.callback_period
verbosity       = C.verbosity

if not os.path.exists(save_filename):
    # Model checkpoint callback
    checkpoint = ModelCheckpoint(
        save_filename,
        monitor='val_acc',
        verbose=verbosity,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        period=callback_period)

    # Fit the model
    history = model.fit(
        x_train, y_train,
        epochs=epochs, batch_size=batch_size, verbose=verbosity,
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
