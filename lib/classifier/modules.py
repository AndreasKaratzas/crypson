
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_channels, out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class Classifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 47):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            ResidualBlock(in_dim, 256),
            ResidualBlock(256, 128),
            ResidualBlock(128, 64),
            nn.Linear(64, num_classes),
        )

    def forward(self, latents):
        return self.features(latents)
