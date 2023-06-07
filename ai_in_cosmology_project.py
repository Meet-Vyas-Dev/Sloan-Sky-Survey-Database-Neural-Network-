# -*- coding: utf-8 -*-
"""AI in cosmology.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uiPZX9p_cS6oLMU-VUZSgugyjhnR0a9r
"""

import pandas as pd 
dataset  = pd.read_csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv")
import numpy as np

x = dataset.drop(columns=["class"])
y = dataset["class"]
print(x)
print(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
print(x_train.shape)
print(y_train.shape)

import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dropout(.05),
        tf.keras.layers.Dense(17, activation='sigmoid'),
        tf.keras.layers.Dropout(.05),
        tf.keras.layers.Dense(17, activation='sigmoid'),
        tf.keras.layers.Dropout(.05),
        tf.keras.layers.Dense(8, activation='sigmoid'),
        tf.keras.layers.Dropout(.05),
        tf.keras.layers.Dense(4, activation='sigmoid'),
        tf.keras.layers.Dropout(.05),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.build()

model.summary()

model.fit(x_train, y_train, steps_per_epoch=1000,epochs =100)

model.evaluate(x_test,y_test)