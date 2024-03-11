
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, num_classes):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        emb = self.label_emb(labels)
        noise = noise.view(noise.size(0), -1)
        gen_input = torch.mul(emb, noise)
        gen_input = gen_input.view(gen_input.size(0), -1, 1, 1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_size, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, img_size*img_size)
        self.model = nn.Sequential(
            nn.Linear(img_size*img_size*2, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_emb = self.label_emb(labels).squeeze()
        d_in = torch.cat((img_flat, label_emb), -1)
        validity = self.model(d_in)
        return validity
