import os
import requests
import zipfile
import io
import pandas as pd

url = 'https://www.dropbox.com/sh/yramhj65py4tsr6/AABhfGJGIagJYpCuxtVzWj_Ua?dl=1'
path = 'data/raw'

def get_data(url=url, path=path, force_download=False, separator=';'):
    """Download and cache the Svebolle data
    
    Parameters
    ----------
    path: string (optional)
        location to save the data
    url: string (optional)
        web location of the data
    force_download: bool (optional)
        if True, force redownloading of data

    Returns
    -------
    data: pandas.DataFrame
        the Svebolle deployment measurement dataset
    """
    if force_download or not os.path.exists(os.path.join(path, 'LORA_data.csv')):
        r = requests.get(url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(path)
    data = pd.read_csv(os.path.join(path, 'LORA_data.csv'), sep=separator)
    return data

def main():
    data = get_data()
    print(data.head())

if __name__ == '__main__':
    main()