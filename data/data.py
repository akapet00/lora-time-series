import os
import requests
import zipfile
import io
import click
import pandas as pd

url = os.environ['SVEBOLLE_URL']
path = 'raw'

@click.command()
@click.argument('url')
@click.argument('path')
def get_data(url=url, path=path, force_download=False, separator=';'):
    """Download and cache the Svebolle data
    
    Parameters\n
    ----------\n
    path: string (optional)\n
        location to save the data\n
    url: string (optional)\n
        web location of the data\n
    force_download: bool (optional)\n
        if True, force redownloading of data\n

    Returns\n
    -------\n
    data: pandas.DataFrame\n
        the Svebolle deployment measurement dataset\n
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