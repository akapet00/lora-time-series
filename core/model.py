from core.utils import Timer

import os
import datetime as dt
import math
import pandas as pd
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error

# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=24, inter_op_parallelism_threads=24)))

class Model():
    """A class for building and inferencing LSTM model"""

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
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
                
        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])
        
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
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]  

        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=2, shuffle=False)
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)
        timer.stop()
    
    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

        