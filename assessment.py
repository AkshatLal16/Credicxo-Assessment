import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
musk_csv = pd.read_csv('musk_csv.csv')

# Labelling the data
X = musk_csv.iloc[:, 3:-1]
y = musk_csv.iloc[:, -1]

# Splitting the dataset into training set and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# Building the neural network
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_shape = (166,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(X_train, y_train, epochs = 100, verbose = 2, validation_data = (X_test, y_test))

# Plotting the graph of training and validation loss
fig = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'],label='train')
plt.xlabel('epochs')
plt.plot(history.history['val_loss'],label='test')
plt.ylabel('loss')
plt.legend()
plt.show()
fig.savefig('loss.jpg')

# Plotting the graph of training and validation accuracy
fig = plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'],label='train')
plt.xlabel('epochs')
plt.plot(history.history['val_accuracy'],label='test')
plt.ylabel('accuracy')
plt.legend()
plt.show()
fig.savefig('accuracy.jpg')

# Predicting the result
y_pred = np.round(model.predict(X_test))

# Calculating the validation performance
from sklearn.metrics import *
ac = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test)
ls = log_loss(y_pred, y_test)
ps = precision_score(y_pred, y_test)
rs = recall_score(y_pred, y_test)

print('Validation Accuracy : ',ac)
print('Validation Loss : ',ls)
print('Precision Score : ',ps)
print('F1 Score : ',f1)
print('Recall Score : ',rs)

# Saving the wrights
model.save_weights('model.h5')
