import torch
import numpy as np

def smile_to_ohe(sample, stoi, pad=120):

    # input is list of strings

    charset_len = len(stoi)

    arr = np.zeros((1, pad), dtype=np.int)
    arr[0,:len(sample)] = np.array([stoi[c] for c in sample], dtype=np.int)

    ohe = np.eye(charset_len)[arr]

    return ohe.astype(np.float32)

def labels_to_smiles(samples, itos):

    # input is numpy arr

    arr = np.vectorize(itos.get)(samples).tolist()

    smiles = []
    for sample in arr:
        out = ''.join([x if x != '<pad>' else ' ' for x in sample])
        smiles.append(out)

    return smiles


