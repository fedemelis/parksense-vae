# ParkSense Variational Autoencoder

This repository contains the code for the Variational Autoencoder designed for the ParkSense project.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Explanation & Major Features

The Variational Autoencoder is a generative model that learns the distribution of the data and can generate new samples from it

The model is trained on the time series data of the parking lots occupancy, and it is able to generate new samples of the
occupancy of the parking lots and also to evaluate the likelihood of a given sample.

## Architecture

The model is composed of an encoder and a decoder. The encoder is a feedforward neural network that takes as input the
time series data and outputs the mean and the log-variance of the latent space. The decoder is another feedforward neural
network that takes as input the latent space and outputs the reconstructed time series data.

The model is trained using the ELBO loss function, which is the sum of the reconstruction loss and the KL divergence loss.

## Purpose

The purpose of this model is to obtain the likelihood of a given time series data, in such a way that we can detect anomalies
in the occupancy of the parking lots.

## Usage

To train the model, run the following command:

```shell
python main.py
```

It will handle your current device and train the model on the data in the `data/time_series.csv` file.
This means that if you train the model on a GPU, it will save the model to a file (vae-cuda.pth) and if you train it on a CPU,
it will save the model to a file (vae-cpu.pth). Note that the model trained on GPU cannot be loaded on CPU and vice versa.
