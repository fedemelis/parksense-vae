import pickle
import torch
import numpy as np
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.optim
from sympy import exp
from torch.utils.data import DataLoader
import pandas as pd
import torch.multiprocessing as mp
import os
from torchviz import make_dot
import datetime

mp.set_start_method('spawn', force=True)
matplotlib.use('TkAgg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):

    """
    Variational Autoencoder

    Args:
    - input_dim: int, input dimension
    - hidden_dim: int, hidden dimension
    - latent_dim: int, latent dimension
    - device: torch.device, device to run the model

    Attributes:
    - encoder: nn.Sequential, encoder network
    - decoder: nn.Sequential, decoder network
    - mean_layer: nn.Linear, mean layer
    - logvar_layer: nn.Linear, logvar layer

    Methods:
    - encode: encode input data
    """
    def __init__(self, input_dim=144, hidden_dim=120, latent_dim=80, device=device):
        super(VAE, self).__init__()

        # encoder
        self.device = device
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )

        # latent mean and variance
        self.mean_layer = nn.Linear(latent_dim, hidden_dim)
        self.logvar_layer = nn.Linear(latent_dim, hidden_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


def likelihood_function(model : VAE, example):
    mean, logvar = model.encode(example)
    z = model.reparameterization(mean, logvar)
    x_hat = model.decode(z)
    rec_loss = nn.functional.binary_cross_entropy(example, x_hat, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    log_likelihood = -((rec_loss + kl_loss) / example.shape[1])
    likelihood = torch.exp(log_likelihood)
    return likelihood.item(), x_hat


def train(model, optimizer, device, train_loader, val_loader, epochs=50, x_dim=144, batch_size=32, patience=8):
    lowest_loss = np.inf
    pat = 0
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            if x.shape[0] == batch_size:
                x = x.view(batch_size, x_dim).to(device)

                optimizer.zero_grad()

                x_hat, mean, log_var = model(x)
                loss = loss_function(x, x_hat, mean, log_var)

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()


        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
        for b_id, x in enumerate(val_loader):
            if x.shape[0] != batch_size:
                continue
            x = x.view(batch_size, x_dim).to(device)
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
        print("\tEpoch", epoch + 1, "\tValidation Loss: ", overall_loss / (b_id * batch_size))
        if (overall_loss / (b_id * batch_size)) < lowest_loss:
            lowest_loss = overall_loss / (b_id * batch_size)
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            print("Early stopping")
            break

    return overall_loss



data_path = 'data/time_series.csv'


def generate_series(mean, var):
    for i in range(25):
        mean_tensor = torch.tensor([[mean]], dtype=torch.float).to(device)
        var_tensor = torch.tensor([[var]], dtype=torch.float).to(device)

        # Usa la funzione di reparametrizzazione del modello per ottenere un campione latente
        z_sample = model.reparameterization(mean_tensor, var_tensor)
        # print("Z sample: ", z_sample.shape)
        # z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)
        # print("Z sample: ", z_sample.shape)
        x_decoded = model.decode(z_sample)
        series = x_decoded.detach().cpu().reshape(144, -1)
        plt.plot(series)
        plt.savefig('images/series_{}.png'.format(i))
        plt.close()


if __name__ == '__main__':

    print("Device: ", device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    data = pd.read_csv(data_path)
    data['ds'] = pd.to_datetime(data['ds'])
    data['ds'] = data['ds'].astype('int64') // 10 ** 9

    grouped = data.groupby('id')

    rows = []
    seq_length = 144

    for name, group in grouped:
        group = group.reset_index(drop=True)
        num_slices = len(group) // seq_length

        for i in range(num_slices):
            slice_y = group['y'].iloc[i * seq_length: (i + 1) * seq_length].to_list()
            rows.append(slice_y)

    new_data = pd.DataFrame(rows)
    new_data.to_csv('data/y_sequences.csv', index=False, header=False)

    dataset = torch.tensor(new_data.values, dtype=torch.float32, device=device)
    tr, te = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])
    dataloader = DataLoader(tr, batch_size=128, shuffle=True)
    dataloader_test = DataLoader(te, batch_size=128, shuffle=True)

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    path = f'models/vae-{device}.pth'
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        train(model, optimizer, device, dataloader, dataloader_test, epochs=10000, x_dim=144, batch_size=128, patience=8)
        torch.save(model.state_dict(), path)


    for i in range(20):
        example = te[i].view(1, 144).to(device)
        likelihood, x_hat = likelihood_function(model, example)
        print("Likelihood: ", likelihood)
        plt.plot(x_hat.detach().cpu().reshape(144, -1))
        plt.plot(example.detach().cpu().reshape(144, -1))
        plt.show()
    #generate_series([0, 0], [1, 1])
