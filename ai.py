import pandas as pd 
import tensorflow as tf

dataset  = pd.read_csv("Skyserver_SQL2_27_2018 6_51_39 PM.csv")

x = dataset.drop(columns=["class"])
y = dataset["class"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential()

model.add(tf.keras.Input(shape=(None, 8000, 17)))
model.add(tf.keras.layers.Dense(8, input_shape=x_train.shape, activation='sigmoid'))
model.add(tf.keras.layers.Dense(8, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)