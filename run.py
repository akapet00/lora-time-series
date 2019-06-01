#! /usr/bin/env python

from core.data_processing import (get_features, clean_features, select_dev, 
                                add_label, datetime_resampling, timeDelay, split)
from core.model import Model

import os
import json
import time 
import math 
import pandas as pd
import numpy as np

def main():
    # importing raw dataset
    PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    DATASET_PATH = 'lora-time-series/data/raw'
    PROCESSED_PATH = 'lora-time-series/data/processed'
    EXTERNALS_PATH = 'lora-time-series/data/externals'

    df = pd.read_csv(os.path.join(PROJ_ROOT, DATASET_PATH, 'LORA_data.csv'), header=0, sep=';')

    configs = json.load(open('configs.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    
    print('DATA importing...')
    data = df.copy()
    print('DATA processing...')
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

    print('DATA creating delays...')
    X, y = timeDelay(df.Active.values, delay=configs['model']['layers'][0]['input_timesteps'])
    # save X and y
    np.save(os.path.join(PROJ_ROOT, EXTERNALS_PATH, 'X.npy'), X)
    np.save(os.path.join(PROJ_ROOT, EXTERNALS_PATH, 'y.npy'), y)

    X_train, y_train, X_test, y_test = split(X, y, ratio=0.8)

    # instantiating a lstm model
    model = Model()
    model.build_model(configs)

    model.train(
		X_train,
	    y_train,
		epochs = configs['training']['epochs'],
		batch_size = configs['training']['batch_size'],
		save_dir = configs['model']['save_dir'],
        logs = configs['model']['logs']
	)

if __name__ == '__main__':
    main()