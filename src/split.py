import pandas as pd 
import numpy as np 

def timeDelay(data, delay):
    X_data, y_data = [], []
    
    for i in range(delay, len(data)):
        X_data.append(data[i-delay: i].tolist())
    X_data = np.array(X_data)
    y_data = data[delay:]
    return np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1)),\
           np.reshape(y_data, (len(y_data), ))

def split(X, y, ratio):    
    test_split = int(len(X) * ratio)
    
    X_train, y_train = X[:test_split], y[:test_split]
    X_test, y_test = X[test_split:], y[test_split:]
    return X_train, y_train, X_test, y_test