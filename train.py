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
import time
from utils import smile_to_ohe, labels_to_smiles

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
    parser.add_argument('-l', '--lr', type=float, default=1e-3,
            help='learning rate (default: 1e-3)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
            help='number of epochs (default: 10)')

    return parser.parse_args()


def setup_logger(name):

    logging.basicConfig(filename=f'./logs/{name}.log',
                        format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    logging.getLogger().addHandler(logging.StreamHandler())


def save_checkpoint(model, optimizer, name, epoch, delete=True):

    # delete last
    if delete:
        # first run will be empty
        try:
            last_check = list(Path('./checkpoints').glob(f'{name}*.pt'))[0]
            last_check.unlink()
        except:
            pass

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, valid_loader, dataset


def setup_model(charset_len, lr, checkpoint=None):

    encoder = Encoder(charset_len)
    decoder = Decoder(charset_len)
    prop = propPred()

    model = ChemVAE(encoder, decoder, prop)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = 0

    # restart training
    if checkpoint is not None:
        cp = torch.load(checkpoint)
        model.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        epoch = cp['epoch']


    return model, optimizer, epoch


def calc_vae_loss(true_labels, x_recon_ohe, z_mu, z_logvar):

    recon_loss = F.cross_entropy(x_recon_ohe, true_labels, reduction='sum')# / true_labels.shape[0]
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu**2 - torch.exp(z_logvar))# / z_mu.shape[0]

    return recon_loss + kl_loss


def train_one_epoch(model, optimizer, loader):

    model.train()

    epoch_loss = 0
    epoch_acc = 0

    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for i, (ohes, labels, props) in loop:

        ohes = ohes.to(device)
        labels = labels.to(device).long()
        props = props.to(device)

        # (b, seq_len, features) -> (b, features, seq_len) for conv
        ohes = ohes.permute(0, 2, 1)

        # encoder out
        # reconstruction through decoder
        # prop prediction from latent
        z_mu, z_logvar, x_recon_ohe, props_out = model(ohes)

        # add all losses and backward
        # crossentropy with ohe input and labeled targets
        # https://discuss.pytorch.org/t/pytorch-lstm-target-dimension-in-calculating-cross-entropy-loss/30398
        vae_loss = calc_vae_loss(labels, x_recon_ohe.permute(0, 2, 1), z_mu, z_logvar)
        prop_loss = F.mse_loss(props_out, props)

        loss = vae_loss + prop_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # reconstruction accuracy, per token over all non-padding
        pred_labels = x_recon_ohe.argmax(dim=-1)
        mask = (labels != 0)
        correct = (torch.masked_select(pred_labels, mask) == torch.masked_select(labels, mask)).sum().float()
        acc = correct / mask.sum()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def run_eval(model, loader):

    # same as train with no optimizer

    model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():

        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        for i, (ohes, labels, props) in loop:

            ohes = ohes.to(device)
            labels = labels.to(device).long()
            props = props.to(device)

            ohes = ohes.permute(0, 2, 1)
            z_mu, z_logvar, x_recon_ohe, props_out = model(ohes)

            vae_loss = calc_vae_loss(labels, x_recon_ohe.permute(0, 2, 1), z_mu, z_logvar)
            prop_loss = F.mse_loss(props_out, props)
            loss = vae_loss + prop_loss

            pred_labels = x_recon_ohe.argmax(dim=-1)
            mask = (labels != 0)
            correct = (torch.masked_select(pred_labels, mask) == torch.masked_select(labels, mask)).sum().float()
            acc = correct / mask.sum()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


def train_n_epochs(n_epochs, model, optimizer, train_loader, valid_loader, start_epoch, name, dataset_stoi):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    start = time.time()
    for epoch in range(1, n_epochs+1):

        check_prog(model, dataset)
        total = start_epoch + epoch

        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader)
        valid_loss, valid_acc = run_eval(model, valid_loader)
        scheduler.step(valid_loss)

        end = time.time()
        elapsed = (end - start) / 60

        if (epoch % 5 == 0) or (epoch == n_epochs):
            save_checkpoint(model, optimizer, name, total, delete=True)

        logging.info(f'Epoch: {total} | Time: {elapsed:.2f}m')
        logging.info(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        logging.info(f'\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%')



def check_prog(model, dataset):

    model.eval()
    with torch.no_grad():

        # fixed examples
        samples = ['COc1ccc(C(=O)N(C)[C@@H](C)C/C(N)=N/O)cc1O',
                   'Cc1ccc(C)c(-n2c(SCCCCCO)nc3ccccc3c2=O)c1',
                   'Fc1ccc(F)c(C[NH+]2CCC(n3cc(-c4cccnc4)nn3)CC2)c1F',
                   'N#Cc1ccc(OC2CCC(NC(=O)c3ccc[nH]3)CC2)nc1',
                   'S=C1[NH+]=N[C@@H]2c3c(sc4c3CC[NH+](Cc3ccccc3)C4)-n3c(n[nH]c3=S)N12']

        ohes = np.concatenate([smile_to_ohe(sample, dataset.stoi) for sample in samples], axis=0)
        ohes = torch.tensor(ohes).to(device).permute(0, 2, 1)

        # push through model and compare reconstruction
        z_mu, z_logvar, x_recon_ohe, props_out = model(ohes)

        preds = x_recon_ohe.argmax(dim=-1).cpu().numpy()
        smiles_out = labels_to_smiles(preds, dataset.itos)

        for i in range(len(samples)):
            logging.info(samples[i])
            logging.info(smiles_out[i])


if __name__ == '__main__':

    args = parse_args()
    setup_logger(args.name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(args)
    logging.info(device)

    train_loader, valid_loader, dataset = setup_loaders(args.valid, args.path, args.batch)
    model, optimizer, epoch = setup_model(len(dataset.itos), args.lr, args.checkpoint)

    logging.info(model)

    train_n_epochs(args.epochs, model, optimizer, train_loader, valid_loader, epoch, args.name, dataset.stoi)
