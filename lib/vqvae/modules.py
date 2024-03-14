
import torch
import numpy as np
import torch.nn as nn

from vector_quantize_pytorch import LFQ


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                 out_channels, num_residual_layers):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels,
                               kernel_size=4, stride=2, padding=1)
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels,
                          kernel_size=3, stride=1, padding=1)
            ) for _ in range(num_residual_layers)
        ])
        self.final_conv = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3,
            stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for res_layer in self.res_layers:
            res = res_layer(x)
            x = res + x
        x = self.final_conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                 out_channels, num_residual_layers):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels,
                               kernel_size=3, stride=1, padding=1)
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels,
                          kernel_size=3, stride=1, padding=1)
            ) for _ in range(num_residual_layers)
        ])
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels, hidden_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 
                      kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        for res_layer in self.res_layers:
            res = res_layer(x)
            x = res + x
        x = self.final_conv(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, 
                 num_residual_layers, num_classes, 
                 latent_dim, num_codebooks, 
                 codebook_size=None,):
        super(AutoEncoder, self).__init__()

        if codebook_size is None:
            codebook_size = 2 ** np.ceil(np.log2(num_classes))

        self.encode = Encoder(
            in_channels, hidden_channels, latent_dim, num_residual_layers)
        self.quantize = LFQ(codebook_size=codebook_size,
                            dim=latent_dim, num_codebooks=num_codebooks)
        self.decode = Decoder(latent_dim, hidden_channels,
                              in_channels, num_residual_layers)

    def forward(self, x):
        # print(f"In AutoEncoder.forward: x.shape = {x.shape}")
        x = self.encode(x)
        x, indices, entropy_aux_loss = self.quantize(x)
        x = self.decode(x)
        return x.clamp(-1, 1), indices, entropy_aux_loss
