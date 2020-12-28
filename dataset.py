import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

class Zinc(Dataset):

    def __init__(self, path):

        self.itos = json.loads(Path(f'{path}/itos.json').read_text())
        self.stoi = json.loads(Path(f'{path}/stoi.json').read_text())

        # convert to int
        self.itos = {int(i): c for i, c in self.itos.items()}
        self.stoi = {c: int(i) for c, i in self.stoi.items()}

        # not memory efficient to preprocess and hold full arrays
        # with np.load(f'{path}/smiles_ohe.npz') as data:
        #     self.smiles_ohe = data['arr_0'].astype(np.float32)

        with np.load(f'{path}/smiles_labeled.npz') as data:
            self.smiles_labeled = data['arr_0'].astype(np.float32)

        self.df = pd.read_csv(f'{path}/250k_rndm_zinc_drugs_clean_3.csv')[['logP', 'qed', 'SAS']].astype(np.float32)

    def to_ohe(self, labels):

        charset_len = len(self.itos)
        return np.eye(charset_len)[labels]

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        labels = self.smiles_labeled[idx, :].astype(np.int)
        props = self.df.iloc[idx, :].values
        ohes = self.to_ohe(labels).astype(np.float32)

        return ohes, labels, props

if __name__ == '__main__':

    dataset = Zinc('./data/zinc')
    print(len(dataset))

    ohes, labels, props = dataset[0]
    print(ohes.shape)
    print(labels.shape)
    print(props.shape)


