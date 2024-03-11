
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size, hidden_dim=256):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_classes, latent_dim)

        self.fc = nn.Linear(latent_dim, 1024)
        self.bn_fc = nn.BatchNorm1d(1024)

        self.res_block1 = ResidualBlock(1024, 1024)
        self.res_block2 = ResidualBlockWShortcut(1024, 512)
        self.res_block3 = ResidualBlockWShortcut(512, hidden_dim)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, z, labels):
        embedding = self.embedding(labels)
        z = torch.mul(z, embedding)
        x = self.fc(z)
        x = self.bn_fc(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        img = self.final_layer(x)
        img = img.view(img.size(0), self.img_size * self.img_size)
        return img


class ResidualBlockWShortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockWShortcut, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels)
        self.bn_shortcut = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        residual = self.shortcut(residual)
        residual = self.bn_shortcut(residual)
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x += residual
        x = nn.LeakyReLU(0.2)(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(0.2)(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x += residual
        x = nn.LeakyReLU(0.2)(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.embedding = nn.Embedding(num_classes, img_size * img_size)
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img = img.view(img.size(0), -1)
        embedding = self.embedding(labels)
        x = torch.cat([img, embedding], dim=1)
        validity = self.model(x)
        return validity
