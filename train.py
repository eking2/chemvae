import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from models import Encoder, Decoder, propPred, ChemVAE
from dataset import Zinc
import logging
from pathlib import Path
from tqdm.auto import tqdm

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch', type=int, default=256,
            help='batch size (default: 256')
    parser.add_argument('-p', '--path', type=str, default='./data/zinc/',
            help='path to data (default: "./data/zinc")')
    parser.add_argument('-v', '--valid', type=float, default=0.1,
            help='valid ratio (default: 0.1')
    parser.add_argument('-c', '--checkpoint', type=str,
            help='checkpoint path to restart training from, optional')
    parser.add_argument('-n', '--name', type=str, required=True,
            help='logging run name')
    parser.add_argument('-l', '--lr', type=float, default=5e-4,
            help='learning rate (default: 3e-4)')
    parser.add_argument('-e', '--epochs', type=int, default=50,
            help='number of epochs (default: 50)')

    return parser.parse_args()


def setup_logger(name):

    logging.basicConfig(filename=f'./logs/{name}.log',
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    logging.getLogger().addHandler(logging.StreamHandler())


def save_checkpoint(model, optimizer, name, epoch, delete=True):

    # delete last
    if delete:
        last_check = list(Path('./checkpoints').glob('*.pt'))[0]
        last_check.unlink()

    torch.save({'epoch' : epoch,
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict()},
        f'./checkpoints/{name}_{epoch}.pt')


def setup_loaders(valid_ratio, path, batch_size):

    dataset = Zinc(path)

    # split into train and valid
    n_samples = len(dataset)
    idx = np.arange(n_samples)
    train_samples = int((1 - valid_ratio) * n_samples)

    train = idx[:train_samples]
    valid = idx[train_samples:]

    train_dataset = Subset(dataset, train)
    valid_dataset = Subset(dataset, valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, dataset


def setup_model(charset_len, lr, checkpoint=None):

    encoder = Encoder(charset_len)
    decoder = Decoder(charset_len)
    prop = propPred()

    model = ChemVAE(encoder, decoder, prop)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = 0

    # restart training
    if checkpoint is not None:
        cp = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    model = model.to(device)

    return model, optimizer, epoch


def calc_vae_loss(true_labels, x_recon_ohe, z_mu, z_logvar):

    recon_loss = F.cross_entropy(x_recon_ohe, true_labels, ignore_index=0, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu**2 - torch.exp(z_logvar))

    return (recon_loss + kl_loss) / len(true_labels)


def train_one_epoch(model, optimizer, loader):

    model.train()

    epoch_loss = 0

    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (ohes, labels, props) in loop:

        ohes = ohes.to(device)
        labels = labels.to(device).long()
        props = props.to(device)

        # (b, seq_len, features) -> (b, features, seq_len) for conv
        # change back for loss
        ohes = ohes.permute(0, 2, 1)

        # encoder out
        # reconstruction through decoder
        # prop prediction from latent
        z_mu, z_logvar, x_recon_ohe, props_out = model(ohes)

        # add all losses and backward
        vae_loss = calc_vae_loss(labels, x_recon_ohe.permute(0, 2, 1), z_mu, z_logvar)
        prop_loss = F.mse_loss(props_out, props)

        loss = vae_loss + prop_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def run_eval(model, loader):

    pass


if __name__ == '__main__':

    args = parse_args()
    setup_logger(args.name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(args)
    logging.info(device)

    train_loader, valid_loader, dataset = setup_loaders(args.valid, args.path, args.batch)
    model, optimizer, epoch = setup_model(len(dataset.itos), args.lr, args.checkpoint)
    train_loss = train_one_epoch(model, optimizer, valid_loader)
    print(train_loss)
