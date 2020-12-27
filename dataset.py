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

        with np.load(f'{path}/smiles_ohe.npz') as data:
            self.smiles_ohe = data['arr_0'].astype(np.float32)

        with np.load(f'{path}/smiles_labeled.npz') as data:
            self.smiles_labeled = data['arr_0'].astype(np.float32)

        self.df = pd.read_csv(f'{path}/250k_rndm_zinc_drugs_clean_3.csv')[['logP', 'qed', 'SAS']].astype(np.float32)

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        # not memory efficient to preprocess and hold full arrays
        # next time read in raw smiles?
        ohes = self.smiles_ohe[idx, :]
        labels = self.smiles_labeled[idx, :]
        props = self.df.iloc[idx, :].values

        return ohes, labels, props
