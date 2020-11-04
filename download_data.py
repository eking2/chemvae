from pathlib import Path
from tqdm.auto import tqdm
import argparse
import requests

data_urls = {'chembl' : 'ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_22/archived/chembl_22_chemreps.txt.gz',
             'zinc' : 'https://github.com/aspuru-guzik-group/chemical_vae/blob/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, choices=['chembl', 'zinc'],
                        help='Dataset to download')

    return parser.parse_args()


def check_file_downloaded(dataset):

    '''check if dataset is already downloaded'''

    file_out = data_urls[dataset].split('/')[-1]
    out_path = Path(f'./data/{file_out}')
    if out_path.exists():
        print(f'Already downloaded {dataset}')
        return True
    return False



def download_ftp_data(dataset):

    pass


def download_http_data(dataset):

    '''download dataset from http link'''

    #headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'}

    if check_file_downloaded(dataset):
        return

    url = data_urls[dataset]
    file_out = data_urls[dataset].split('/')[-1]
    file_path = f'./data/{file_out}'

    r = requests.get(url, stream=True) 
    r.raise_for_status()
    total_size = int(r.headers.get('content-length'))

    with open(file_path, 'wb') as fi, tqdm(desc=base, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in r.iter_content(chunk_size = 4096):
            size = fi.write(data)
            bar.update(size)
    

def convert_to_hd5(dataset):

    pass
    


if __name__ == '__main__':

    args = parse_args()
    download_http_data(args.dataset)
