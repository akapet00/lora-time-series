import os
import pandas as pd 
import numpy as np

def get_features(df):
    return df[['Time', 'DevAddr']]

def clean_features(df):
    Time = list(df.Time.values)
    Time_parsed = []
    sep = '.'
    for t in Time:
        t_parsed = t.split(sep, 1)[0]
        Time_parsed.append(t_parsed)
    df.Time = Time_parsed
    df['Time'] = pd.to_datetime(df['Time'])
    return df

def select_dev(df, dev_name='000013c1'):
    df = df[df.DevAddr == dev_name]
    df = df.reset_index(drop=True)
    df = df.Time
    df = df.drop_duplicates(keep='last')
    df = pd.DataFrame({'Time':df.values})
    return df

def add_label(df):
    df['Active'] = 1
    return df

def datetime_resampling(df):
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index)
    df = df.resample('1S').asfreq()
    df = df.fillna(0)
    return df

def timeDelay(df, delay):
    X_data, y_data = [], []
    
    for i in range(delay, len(df)):
        X_data.append(df[i-delay: i].tolist())
    X_data = np.array(X_data)
    y_data = df[delay:]
    return np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1)),\
           np.reshape(y_data, (len(y_data), ))

def split(X, y, ratio):    
    test_split = int(len(X) * ratio)
    
    X_train, y_train = X[:test_split], y[:test_split]
    X_test, y_test = X[test_split:], y[test_split:]
    return X_train, y_train, X_test, y_test

def main():
    pass

if __name__ == '__main__':
    main()