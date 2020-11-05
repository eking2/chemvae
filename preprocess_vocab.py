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

        self.charset = list({char for smile in self.smiles for char in smile})

    def to_labels(self):

        # 0 = pad
        self.itos = dict(enumerate(self.charset, 1))
        self.itos[0] = '<pad>'

        self.stoi = {char : idx for idx, char in self.itos.items()}

        self.labeled = np.zeros((len(self.smiles), self.pad))
        for i, smile in enumerate(self.smiles):
            smile_to_label = np.array([self.stoi[char] for char in smile])
            self.labeled[i,:len(smile_to_label)] = smile_to_label

    def labels_to_onehot(self):

        pass

    def save_processed(self):

        pass

    def run_preprocess(self):

        self.get_charset()
        self.to_labels()
        self.labels_to_onehot()




def parse_args():


    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['chembl', 'zinc'])

    return parser.parse_args()

def preprocess_chembl():

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
    smiles = df['canonical_smiles']


    # to one hot


def preprocess_zinc():

    # cols = ['smiles', 'logP', 'qed', 'SAS']
    df = pd.read_csv('./data/zinc/250k_rndm_zinc_drugs_clean_3.csv')
    smiles = df['smiles'].str.strip().to_numpy()

    featurizer = smiles_to_onehot(smiles)
    featurizer.run_preprocess()


if __name__ == '__main__':

    args = parse_args()

    if args.dataset == 'chembl':
        preprocess_chembl()

    elif args.dataset == 'zinc':
        preprocess_zinc()
