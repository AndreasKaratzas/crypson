# https://medium.com/@outerrencedl/variational-autoencoder-and-a-bit-kl-divergence-with-pytorch-ce04fd55d0d7

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, num_layers, img_size=32):
        super(Encoder, self).__init__()

        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
            encoder_layers.append(nn.BatchNorm2d(out_channels))
            encoder_layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= 2
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(
            out_channels // 2 * (img_size // 2 ** num_layers) ** 2, latent_dim)
        self.fc_log_var = nn.Linear(
            out_channels // 2 * (img_size // 2 ** num_layers) ** 2, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var
    

class Decoder(nn.Module):
    def __init__(self, latent_dim, img_size, out_channels, num_layers):
        super(Decoder, self).__init__()
        
        decoder_layers = []
        out_channels = out_channels // 2
        decoder_layers.append(
            nn.Linear(latent_dim, out_channels * (img_size // 2 ** num_layers) ** 2))
        decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Unflatten(
            1, (out_channels, img_size // 2 ** num_layers, img_size // 2 ** num_layers)))
        for _ in range(num_layers):
            decoder_layers.append(nn.ConvTranspose2d(
                out_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1))
            decoder_layers.append(nn.BatchNorm2d(out_channels // 2))
            decoder_layers.append(nn.ReLU())
            out_channels //= 2
        decoder_layers.append(
            nn.Conv2d(out_channels, 1, kernel_size=3, padding=1))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, latent_dim, img_size=32):
        super(AutoEncoder, self).__init__()

        self.encode = Encoder(in_channels=in_channels, out_channels=hidden_channels,
                              latent_dim=latent_dim, num_layers=num_layers, img_size=img_size)
        self.decode = Decoder(latent_dim=latent_dim, img_size=img_size,
                              out_channels=hidden_channels, num_layers=num_layers)

        self._confirm_functionality(torch.randn(1, in_channels, img_size, img_size))

    def _confirm_functionality(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)

        if not out.shape == x.shape:
            raise ValueError(f'Output shape {out.shape} does not match input shape {x.shape}')

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
