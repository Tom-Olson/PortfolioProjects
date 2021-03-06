# -*- coding: utf-8 -*-
"""CS767_Thomas_Olson_Assignment_5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ur3ZZF_zltppvX8J_2oJlcN76liTP6M8
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from functools import partial
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint


print(tf.__version__)
print(keras.__version__)

"""# **Question 2:**"""

# loading and splitting the dataset
fashion_mnist = keras.datasets.fashion_mnist

(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()

#min-max normalization - min is 0 so just divide by max
X_train_full = X_train_full/255.0
X_test = X_test / 255.0

class_names = ["T-shirt/top", "Trouser", "pullover", "Dress", "Coat", 
				"Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, Y_train_full, test_size=0.1,random_state=42)

"""### Part 1"""

class DNN():


  # Define the Keras model
  def get_model(self,num_hidden_layers=1):
    network = keras.models.Sequential()
    network.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n in range(num_hidden_layers):
      network.add(keras.layers.Dense(100, activation="relu"))
    network.add(keras.layers.Dense(10, activation="softmax"))
    network.compile(loss='sparse_categorical_crossentropy', # Cross-entropy
                optimizer="sgd", # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric
    return network

#call model with just one hidden layer
keras.backend.clear_session()
test = DNN()
network = test.get_model(1)

history = network.fit(X_train , y_train , epochs = 10,
	validation_data=(X_valid,y_valid))                      # Validation optional. 


history.params
history.epoch
history.history.keys()
history.history.get('accuracy') 		# Get any keys <function dict.get(key, default=None, /)>


# Plot the learing curve 
import pandas as pd
import matplotlib.pyplot as plt
# history.history is a dictionary containing the loss and measurements. 
# Change it to DataFrame, then method plot() can be used to get the learning curves.
print(pd.DataFrame(history.history))

pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # Set the virtical range to [0,1]
plt.show()

score = network.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

"""### Part 2"""

#call model with 2 hidden layers
keras.backend.clear_session()
test = DNN()
network = test.get_model(2)

history = network.fit(X_train , y_train , epochs = 10,
	validation_data=(X_valid,y_valid))                        # Validation optional. 


history.params
history.epoch
history.history.keys()
history.history.get('accuracy') 		# Get any keys <function dict.get(key, default=None, /)>


# Plot the learing curve 
import pandas as pd
import matplotlib.pyplot as plt
# history.history is a dictionary containing the loss and measurements. 
# Change it to DataFrame, then method plot() can be used to get the learning curves.
print(pd.DataFrame(history.history))

pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # Set the virtical range to [0,1]
plt.show()

score = network.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

"""### Part 3"""

keras.backend.clear_session()
test = DNN()
network = test.get_model(3)

history = network.fit(X_train , y_train , epochs = 10,
	validation_data=(X_valid,y_valid))                      # Validation optional. 


history.params
history.epoch
history.history.keys()
history.history.get('accuracy') 		# Get any keys <function dict.get(key, default=None, /)>


# Plot the learing curve 
import pandas as pd
import matplotlib.pyplot as plt
# history.history is a dictionary containing the loss and measurements. 
# Change it to DataFrame, then method plot() can be used to get the learning curves.
print(pd.DataFrame(history.history))

pd.DataFrame(history.history).plot(figsize=(8, 5)) 
plt.grid(True)
plt.gca().set_ylim(0, 1) # Set the virtical range to [0,1]
plt.show()

score = network.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

"""# **Question 3**"""

mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

#only want numbers 0-4
train_mask = np.isin(y_train_full, [0,1,2,3,4])
test_mask = np.isin(y_test, [0,1,2,3,4])

X_train_full, y_train_full = X_train_full[train_mask], y_train_full[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]


X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1,random_state=42)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_valid shape", X_valid.shape)
print("y_valid shape", y_valid.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

#visualize the numbers
import random  
plt.rcParams['figure.figsize'] = (9,9) 

for i in range(9):
    plt.subplot(3,3,i+1)
    num = random.randint(0, len(X_train))
    plt.imshow(X_train[num], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[num]))
    
plt.tight_layout()

#min-max normalization - min is 0 so just divide by max
X_train = X_train/255.0
X_valid = X_valid/255.0
X_test = X_test/255.0

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(5,activation='softmax')
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5") # By default saves the model at the end of each Epoch

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),batch_size=128,
                    callbacks=[checkpoint_cb, early_stopping_cb])

score = model.evaluate(X_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predictions = np.argmax(model.predict(X_test),axis=1)

# Check which items we got right / wrong
correct_indices = np.nonzero(predictions == y_test)[0]

incorrect_indices = np.nonzero(predictions != y_test)[0]

#visualize 9 correct and 9 wrong
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predictions[correct], y_test[correct]))
    
plt.tight_layout()
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predictions[incorrect], y_test[incorrect]))
    
plt.tight_layout()

#Try different parameters
model.compile(loss="sparse_categorical_crossentropy", optimizer='nadam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),batch_size=128,
                    callbacks=[checkpoint_cb, early_stopping_cb])

score = model.evaluate(X_test, y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

"""# Question 4"""

history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])

model.save_weights("myKerasWeights.ckpt")

mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

train_mask = np.isin(y_train_full, [5,6,7,8,9])
test_mask = np.isin(y_test, [5,6,7,8,9])

X_train_full, y_train_full = X_train_full[train_mask], y_train_full[train_mask]
X_test, y_test = X_test[test_mask], y_test[test_mask]

from sklearn.model_selection import StratifiedShuffleSplit

n_splits = 1  # We only want a single split in this case
sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=500, random_state=0)

for train_index, test_index in sss.split(X_train_full, y_train_full):
    X_train, X_test = X_train_full[train_index], X_train_full[test_index]
    y_train, y_test = y_train_full[train_index], y_train_full[test_index]




X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2,random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1,random_state=42)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_valid shape", X_valid.shape)
print("y_valid shape", y_valid.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

import random  
plt.rcParams['figure.figsize'] = (9,9) # Make the figures a bit bigger

for i in range(9):
    plt.subplot(3,3,i+1)
    num = random.randint(0, len(X_train))
    plt.imshow(X_train[num], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[num]))
    
plt.tight_layout()

#min-max normalization - min is 0 so just divide by max
X_train = X_train/255.0
X_valid = X_valid/255.0
X_test = X_test/255.0

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
# Reshape data
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_valid = y_valid.reshape(-1, 1)

# Fit and transform training data
ohe.fit(y_train)
transformed_train = ohe.transform(y_train).toarray()

# Fit and transform testing data
ohe.fit(y_test)
transformed_test = ohe.transform(y_test).toarray()

ohe.fit(y_valid)
transformed_valid = ohe.transform(y_valid).toarray()

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(100, activation="elu"),
    keras.layers.Dense(5,activation='softmax')
])

model.summary()

model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("weights.{epoch:02d}-{val_loss:.2f}.hdf5") # By default saves the model at the end of each Epoch

early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

history = model.fit(X_train, transformed_train, epochs=2)

score = model.evaluate(X_test, transformed_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
predictions = np.argmax(model.predict(X_test),axis=1)

# Check which items we got right / wrong
correct_indices = np.nonzero(predictions == y_test)[0]

incorrect_indices = np.nonzero(predictions != y_test)[0]

plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predictions[correct], y_test[correct]))
    
plt.tight_layout()
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predictions[incorrect], y_test[incorrect]))
    
plt.tight_layout()

keras.backend.clear_session()
model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, transformed_train, epochs=20)

score = model.evaluate(X_test, transformed_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

