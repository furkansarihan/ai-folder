import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

print("Tensorflow Version: " + str(tf.VERSION))
print("Keras Version: " + str(tf.keras.__version__))

model = tf.keras.Sequential([
layers.Dense(64, activation='relu'),
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy, 
              metric=[tf.keras.metrics.categorical_accuracy])

data = np.random.random((10, 8))
#labels = np.random.random((10, 8))

#val_data = np.random.random((1000, 32))
#val_labels = np.random.random((1000, 10))

#model.fit(data, labels, epochs=10, batch_size=32)
