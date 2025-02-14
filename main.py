import pickle
import torch
import numpy as np
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import torch.optim
from torch.utils.data import DataLoader
import pandas as pd
import torch.multiprocessing as mp
import os

mp.set_start_method('spawn', force=True)
matplotlib.use('TkAgg')
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=120, latent_dim=80, device=device):
        super(VAE, self).__init__()

        # encoder
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
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
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


def train(model, optimizer, device, train_loader, epochs=50, x_dim=144, batch_size=32):
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
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.9e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if os.path.exists('models/vae.pth'):
        model.load_state_dict(torch.load('models/vae.pth', weights_only=True))
    else:
        train(model, optimizer, device, dataloader, epochs=100, x_dim=144, batch_size=128)
        torch.save(model.state_dict(), 'models/vae.pth')

    for i in range(110, 150):
        example = dataset[i].view(1, 144)
        x_hat, mean, logvar = model(example)
        loss = loss_function(example, x_hat, mean, logvar)
        print("Loss: ", loss.item())
        plt.plot(x_hat.detach().cpu().reshape(144, -1))
        plt.plot(example.detach().cpu().reshape(144, -1))
        plt.show()

    generate_series([0, 0], [1, 1])
