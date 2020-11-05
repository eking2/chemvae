from pathlib import Path
from tqdm.auto import tqdm
from ftplib import FTP
import argparse
import requests

data_urls = {'chembl' : 'ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_22/archived/chembl_22_chemreps.txt.gz',
             'zinc' : 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc/250k_rndm_zinc_drugs_clean_3.csv'}


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, type=str, choices=['chembl', 'zinc'],
                        help='Dataset to download')

    return parser.parse_args()


def check_file_downloaded(dataset):

    '''check if dataset is already downloaded'''

    file_out = data_urls[dataset].split('/')[-1]
    out_path = Path(f'./data/{dataset}/{file_out}')
    if out_path.exists():
        print(f'\n{dataset} already downloaded\n')
        return True
    return False


def get_url_info(dataset):

    '''file names from url and make output folder'''

    url = data_urls[dataset]

    file_out = data_urls[dataset].split('/')[-1]
    file_path = Path(f'./data/{dataset}')
    name = Path(file_path, file_out)

    if not file_path.exists():
        file_path.mkdir()

    return url, file_out, file_path, name


def download_ftp_data(dataset):

    '''download dataset from ftp'''

    if check_file_downloaded(dataset):
        return

    url, file_out, file_path, name = get_url_info(dataset)

    ftp_host = url.split('/')[2]
    ftp_path = url.split('/', 3)[-1]

    ftp = FTP(host=ftp_host)
    ftp.login()

    total_size = int(ftp.size(ftp_path))

    with open(name, 'wb') as fi, tqdm(desc=file_out, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        def callback_(data):
            bar.update(len(data))
            fi.write(data)

        ftp.retrbinary('RETR {}'.format(ftp_path), callback_)


def download_http_data(dataset):

    '''download dataset from http link'''

    #headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'}

    if check_file_downloaded(dataset):
        return

    url, file_out, file_path, name = get_url_info(dataset)

    r = requests.get(url, stream=True)
    r.raise_for_status()
    total_size = int(r.headers.get('content-length'))

    with open(name, 'wb') as fi, tqdm(desc=file_out, total=total_size, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in r.iter_content(chunk_size = 1024):
            size = fi.write(data)
            bar.update(size)


if __name__ == '__main__':

    args = parse_args()

    if args.dataset == 'zinc':
        download_http_data(args.dataset)
    elif args.dataset == 'chembl':
        download_ftp_data(args.dataset)


