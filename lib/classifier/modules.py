import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class Classifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 47, dropout_rate=0.2):
        super(Classifier, self).__init__()
        self.features = nn.Sequential(
            ResidualBlock(in_dim, 512, dropout_rate),
            ResidualBlock(512, 512, dropout_rate),
            ResidualBlock(512, 256, dropout_rate),
            ResidualBlock(256, 128, dropout_rate),
            ResidualBlock(128, 64, dropout_rate),
            nn.Linear(64, num_classes),
        )

    def forward(self, latents):
        return self.features(latents)
