from split import timeDelay, split

import os
import pandas as pd
import math

import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=3, inter_op_parallelism_threads=3)))

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
PROCESSED_PATH = 'src/data/processed'
df = pd.read_csv(os.path.join(PROJ_ROOT, PROCESSED_PATH, 'single_device_activation.csv'),
                 header=0, sep=',')

X, y = timeDelay(df.Active.values, 5) # train test
ratio = 0.8 # train-test ratio
window_size = 5 # delay

X_train, y_train, X_test, y_test = split(X, y, ratio)

model = Sequential()
model.add(LSTM(50, input_shape=(window_size, 1))) # 50 neurons
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mse', optimizer='adam')

history = model.fit(X_train, y_train, 
                        epochs=20, 
                        batch_size=32, 
                        validation_data=(X_test, y_test), 
                        verbose=2, 
                        shuffle=False)

joblib.dump(history, 'lstm_single_device.pkl')

# generate predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Estimate model performance
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))