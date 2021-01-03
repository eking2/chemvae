import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Repeat(nn.Module):
    def __init__(self, reps):
        super().__init__()
        self.reps = reps

    def forward(self, x):
        # repeat sequence length times
        out = x.unsqueeze(1).repeat(1, self.reps, 1)
        return out


class Encoder(nn.Module):
    def __init__(self, charset):
        super().__init__()

        # out = (in + 2P - (k - 1) - 1)/s + 1

        # input = (batch, charset, seq_len) = zinc (batch, 35, 120)
        self.conv = nn.Sequential(
            nn.Conv1d(charset, 9, kernel_size=9, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(9),
            # (batch, 9, 112)

            nn.Conv1d(9, 9, kernel_size=9, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(9),
            # (batch, 9, 104)

            nn.Conv1d(9, 10, kernel_size=11, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(10),
            # (batch, 10, 94)

            nn.Flatten(),
            # (batch, 940)

            nn.Linear(940, 435),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(435)
            # (batch, 435)
        )

        self.fc11 = nn.Linear(435, 292)
        self.fc12 = nn.Linear(435, 292)

    def forward(self, x):

        out = self.conv(x)

        z_mu = self.fc11(out)
        z_logvar = self.fc12(out)

        return z_mu, z_logvar


class Decoder(nn.Module):
    def __init__(self, charset):
        super().__init__()

        # input = (batch, z_dim) = (batch, 292)
        self.net = nn.Sequential(
            nn.Linear(292, 292, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(292),
            # (batch, 292)

            Repeat(120),
            # (batch, 120, 292)
        )

        self.gru = nn.GRU(292, 501, num_layers=3, batch_first=True)
        # (batch, 120, 501)

        #self.gru_final = nn.GRU(501, charset, batch_first=True)
        self.fc = nn.Linear(501, charset)
        # (batch, 120, 35)

    def forward(self, x):

        out = self.net(x)
        out, hidden = self.gru(out)
        out = self.fc(out)
        #out, hidden = self.gru_final(out)

        return out


class propPred(nn.Module):
    def __init__(self):
        super().__init__()

        # input = (batch, z_dim) = (batch, 292)
        self.net = nn.Sequential(
            nn.Linear(292, 67),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            # (batch, 67)

            nn.Linear(67, 66, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(66),
            # (batch, 66)

            nn.Linear(66, 65, bias=False),
            nn.LeakyReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(65),
            # (batch, 65)

            nn.Linear(65, 3)
            # (batch, 3)
        )

    def forward(self, x):

        return self.net(x)


class ChemVAE(nn.Module):
    def __init__(self, encoder, decoder, prop_pred):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.prop_pred = prop_pred

    def reparameterize(self, z_mu, z_logvar, scale=1e-2):

        std = torch.exp(0.5 * z_logvar)
        eps = scale * torch.randn_like(std)

        return z_mu + eps*std

    def forward(self, x):

        z_mu, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mu, z_logvar)

        decoded = self.decoder(z)
        props = self.prop_pred(z_mu)

        return z_mu, z_logvar, decoded, props


if __name__ == '__main__':

    test_encoder = Encoder(35)
    # transpose seq_len and charset from loader
    x = torch.randn(2, 35, 120)
    encoder_out = test_encoder(x)
    print(f'encoder input: {x.shape}')
    print(f'z_mu: {encoder_out[0].shape}')
    print(f'z_logvar: {encoder_out[1].shape}')
    print()

    test_decoder = Decoder(35)
    x = torch.randn(2, 292)
    decoder_out = test_decoder(x)
    print(f'decoder input: {x.shape}')
    print(f'decoder output: {decoder_out.shape}')
    print()

    test_prop = propPred()
    prop_out = test_prop(x)
    print(f'prop pred input: {x.shape}')
    print(f'prop pred output: {prop_out.shape}')
    print()

    test_chemvae = ChemVAE(test_encoder, test_decoder, test_prop)
    x = torch.randn(2, 35, 120)
    chemvae_out = test_chemvae(x)
    print(f'chemvae input: {x.shape}')
    print(len(chemvae_out))
