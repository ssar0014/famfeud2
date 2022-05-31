import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

if tf.config.list_physical_devices('GPU'):
    print("GPU IS AVAILABLE")

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# checking images shape
print("checking images shape",X_train.shape, X_test.shape)

# display single image shape
print("display single image shape",X_train[0].shape)

# checking labels
print("checking labels",y_train[:5])

# scaling image values between 0-1
X_train_scaled = X_train/255
X_test_scaled = X_test/255

# one hot encoding labels
y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')

def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')    
    ])
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model

# %%timeit -n1 -r1
# CPU
with tf.device('/CPU:0'):
    model_cpu = get_model()
    model_cpu.fit(X_train_scaled, y_train_encoded, epochs = 5)

# %%timeit -n1 -r1
# GPU
with tf.device('/GPU:0'):
    model_gpu = get_model()
    model_gpu.fit(X_train_scaled, y_train_encoded, epochs = 5)

