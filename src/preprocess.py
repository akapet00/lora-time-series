import click
import os
import pandas as pd 

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
    print(df)
    return df

def main():
    PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    DATASET_PATH = 'src/data/raw'
    PROCESSED_PATH = 'src/data/processed'
    data_fname = os.path.join(PROJ_ROOT, DATASET_PATH, 'LORA_data.csv')
    df = pd.read_csv(data_fname, header=0, sep=';')

    data = df.copy()
    data = get_features(data)
    data = clean_features(data)

    dev = select_dev(data, dev_name='000013c1')
    dev = add_label(dev)

    dev_resampled = datetime_resampling(dev)
    dev_resampled.to_csv(os.path.join(PROJ_ROOT, PROCESSED_PATH, 'single_device_activation.csv'), sep=',')

if __name__ == '__main__':
    main()