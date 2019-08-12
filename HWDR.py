from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image
import numpy as np
import os

class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('acc')>0.998):
            print("\n Reached 99.8% accuracy. So cancelling training")
            self.model.stop_training=True

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train / 255         #Normalization - greyscale images contain 255 pixels(speeds up the process)
X_test = X_test / 255

early_stopping_monitor = EarlyStopping(patience=3)
callbacks=myCallBack()

model = tf.keras.Sequential(                                              #Sequential layers
    [tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer = "adam", loss="sparse_categorical_crossentropy", metrics = ["accuracy"])
model.fit(X_train, y_train, epochs = 15, callbacks = [early_stopping_monitor, callbacks])
model.evaluate(x = X_test, y = y_test , verbose = 1)

# To test your model with your own hand-written digit images
img = Image.open('C:\\Users\\user\\Documents\\ML\\MLProjects\\6.png').convert("L")
img = img.resize((28,28))
im2arr = np.array(img)
#print(im2arr)
im2arr = im2arr.reshape(1,28,28,1)
y_pred = model.predict(im2arr)
print(y_pred)
