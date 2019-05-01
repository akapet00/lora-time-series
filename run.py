#! /usr/bin/env python

from core.data_processing import (get_features, clean_features, select_dev, 
                                add_label, datetime_resampling, timeDelay, split)
from core.model import Model

import os
import json
import time 
import math 
import pandas as pd

def main():
    # importing raw dataset
    PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    DATASET_PATH = 'lora-time-series/data/raw'
    PROCESSED_PATH = 'lora-time-series/data/processed'

    df = pd.read_csv(os.path.join(PROJ_ROOT, DATASET_PATH, 'LORA_data.csv'), header=0, sep=';')

    data = df.copy()
    data = get_features(data)
    data = clean_features(data)

    dev = select_dev(data, dev_name='000013c1')
    dev = add_label(dev)

    dev_resampled = datetime_resampling(dev)
    dev_resampled.to_csv(os.path.join(PROJ_ROOT, PROCESSED_PATH, 'single_device_activation.csv'), 
                        sep=',')

    # train-test split on processed dataset
    df = pd.read_csv(os.path.join(PROJ_ROOT, PROCESSED_PATH, 'single_device_activation.csv'),
                    header=0, sep=',')

    X, y = timeDelay(df.Active.values, delay=5)
    X_train, y_train, X_test, y_test = split(X, y, ratio=0.8)

    # instantiating a lstm model
    configs = json.load(open('configs.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

    model = Model()
    model.build_model(configs)

    model.train(
		X_train,
	    y_train,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir']
	)

    # predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    trainScore = model.evaluate(X_train, y_train)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
    testScore = model.evaluate(X_test, y_test)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

if __name__ == '__main__':
    main()