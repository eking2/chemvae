from pathlib import Path
import pandas as pd
import numpy as np
import gzip
import argparse
import json
import shutil


class smiles_to_onehot:

    def __init__(self, smiles, pad=120):

        self.smiles = smiles
        self.pad = pad

        self.charset = None
        self.itos = None
        self.stoi = None
        self.labeled = None

    def get_charset(self):

        '''get all possible characters from smiles'''

        self.charset = list({char for smile in self.smiles for char in smile})

    def to_labels(self):

        '''label encode (numericalize) smile strings, pad up to max length'''

        # 0 = pad
        self.itos = dict(enumerate(self.charset, 1))
        self.itos[0] = '<pad>'

        self.stoi = {char : idx for idx, char in self.itos.items()}

        self.labeled = np.zeros((len(self.smiles), self.pad))
        for i, smile in enumerate(self.smiles):
            smile_to_label = np.array([self.stoi[char] for char in smile])
            self.labeled[i,:len(smile_to_label)] = smile_to_label

    def labels_to_onehot(self):

        '''one hot encode numericalzied smiles'''

        # broadcast compare
        # unsqueeze last dim on labels, boolean compare to range up to max label, TF to 1/0 as int
        # (samples, max seq len, num labels)
        self.ohe = (np.arange(self.labeled.max() + 1) == np.expand_dims(self.labeled, -1)).astype(int)

    def save_processed(self, dataset):

        '''save labels, vocab, ohe encoded'''

        dest_dir = Path(f'./data/{dataset}')

        with open(Path(dest_dir, 'itos.json'), 'w') as fo:
            json.dump(self.itos, fo, indent=4)

        with open(Path(dest_dir, 'stoi.json'), 'w') as fo:
            json.dump(self.stoi, fo, indent=4)

        np.savez_compressed(Path(dest_dir, 'smiles_labeled'), self.labeled)
        np.savez_compressed(Path(dest_dir, 'smiles_ohe'), self.ohe)


    def run_preprocess(self):

        print('getting charset...')
        self.get_charset()

        print('smiles to labels...')
        self.to_labels()
        print(f'number of characters: {len(self.itos)}')
        print(self.itos)
        print(f'smiles labeled: {self.labeled.shape}')

        print('one hot encoding...')
        self.labels_to_onehot()
        print(f'smiles ohe: {self.ohe.shape}')


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['chembl', 'zinc'])
    parser.add_argument('-p', '--pad', type=int, default=120)

    return parser.parse_args()

def preprocess_chembl(pad):

    data = Path('./data/chembl/chembl_22_chemreps.txt.gz')
    assert data.exists(), 'chembl dataset missing'

    data_out = Path('./data/chembl/chembl.txt')

    # unpack
    if not data_out.exists():
        with gzip.open(data, 'rb') as fi:
            with open(data_out, 'wb') as fo:
                shutil.copyfileobj(fi, fo)

    # cols = ['chembl_id', 'canonical_smiles', 'standard_inchi', 'standard_inchi_key']
    df = pd.read_csv('./data/chembl/chembl.txt', delim_whitespace=True)
    smiles = df['canonical_smiles'].str.strip().to_numpy()

    # remove smiles longer than pad
    smiles = [smile for smile in smiles if len(smile) <= pad]

    # to one hot
    featurizer = smiles_to_onehot(smiles, pad)
    featurizer.run_preprocess()
    featurizer.save_processed('chembl')


def preprocess_zinc(pad):

    # cols = ['smiles', 'logP', 'qed', 'SAS']
    df = pd.read_csv('./data/zinc/250k_rndm_zinc_drugs_clean_3.csv')
    smiles = df['smiles'].str.strip().to_numpy()

    smiles = [smile for smile in smiles if len(smile) <= pad]
    print(f'samples: {len(smiles):,}')
    featurizer = smiles_to_onehot(smiles, pad)
    featurizer.run_preprocess()
    featurizer.save_processed('zinc')


if __name__ == '__main__':

    args = parse_args()
    print(args.dataset)

    # ohe too large to fit in memory, use sparse?
    if args.dataset == 'chembl':
        #preprocess_chembl(args.pad)
        pass

    elif args.dataset == 'zinc':
        preprocess_zinc(args.pad)
