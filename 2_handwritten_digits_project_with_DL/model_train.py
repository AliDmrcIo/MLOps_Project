"""
MNIST verisetini kullanıcaz: handwritten digits

CNN ile Classification problemi çözeceğiz

Keras ve tensorflow ile yapacağız işlemlerimizi
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

import numpy as np

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# loading dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalizating dataset - görseller böyle normalize ediliyordu hatırla. çünkü 0-255 arası değer alır ger bir px
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# dimension fixing
X_train = np.expand_dims(X_train, axis=-1) # (60.000, 28, 28, 1)
X_test = np.expand_dims(X_test, axis=-1)   # (60.000, 28, 28, 1)

y_train = to_categorical(y_train, num_classes=10) # y_train'i kategorik hale getiriyorum(float->string) (60.000, 10)
y_test = to_categorical(y_test, num_classes=10)   # y_test'i kategorik hale getiriyorum(float->string) (60.000, 10)

# model create/build
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# compiler
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# model training
model.fit(X_train, y_train, epochs=3, batch_size=32, validation_split=0.2)

# model save
model.save("mnist_model.h5") # ml modellerini kaydederken .pkl, deep learning modellerini kaydederken .h5