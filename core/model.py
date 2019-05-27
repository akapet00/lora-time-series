from core.utils import Timer

import os
import datetime as dt
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense, LSTM, CuDNNLSTM, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import backend as K
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

class Model():
    def __init__(self):
        self.model = Sequential()

    def build_model(self, configs):
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(CuDNNLSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
            if layer['type'] == 'BatchNormalization':
                self.model.add(BatchNormalization())
                
        self.model.compile(loss=configs['model']['loss'], 
                           optimizer=configs['model']['optimizer'],
                           metrics=['accuracy'])

        print('[Model] Model Compiled')
        timer.stop()   

    def train(self, X, y, epochs, batch_size, save_dir):
        timer = Timer() 
        timer.start()
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size'% (epochs, batch_size))

        save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))

        callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True),
            #TensorBoard(log_dir=f"/logs/{epochs}-epochs-{dt.datetime.now().strftime('%d%m%Y-%H%M%S')}")
        ]  

        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2, shuffle=False)
        self.model.save(save_fname)
        timer.stop()
    
    def predict(self, x):
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(x)
        predicted = np.reshape(predicted, (predicted.size,))
        return predicted
    
    def predict_sequences_multiple(self, data, window_size, prediction_len):
		# Predict sequence of 50 steps before shifting prediction run forward by 50 steps
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[np.newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

        