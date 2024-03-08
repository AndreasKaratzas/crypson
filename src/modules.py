import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.embedding = nn.Embedding(num_classes, latent_dim)
        self.init_size = img_size // 8
        self.fc = nn.Linear(latent_dim, 512 * self.init_size ** 2)
        self.res_blocks = nn.ModuleDict({
            'res_block1': ResidualBlock(512, 512, num_classes),
            'attention1': AttentionLayer(512),
            'upsample1': nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            'res_block2': ResidualBlock(256, 256, num_classes),
            'attention2': AttentionLayer(256),
            'upsample2': nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            'res_block3': ResidualBlock(128, 128, num_classes),
            'attention3': AttentionLayer(128),
            'upsample3': nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            'res_block4': ResidualBlock(64, 64, num_classes),
            'attention4': AttentionLayer(64),
            'conv': nn.Conv2d(64, 1, 3, padding=1),
            'tanh': nn.Tanh()
        })

    def forward(self, z, labels):
        embedding = self.embedding(labels)
        z = torch.mul(z, embedding)
        z = self.fc(z)
        z = z.view(z.size(0), 512, self.init_size, self.init_size)

        for name, module in self.res_blocks.items():
            if isinstance(module, ResidualBlock):
                z = module(z, labels)
            else:
                z = module(z)

        img = z
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.embedding = nn.Embedding(num_classes, img_size * img_size)
        self.model = nn.ModuleDict({
            'conv1': nn.utils.spectral_norm(nn.Conv2d(2, 64, 4, stride=2, padding=1)),
            'leaky_relu1': nn.LeakyReLU(0.2),
            'conv2': nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            'leaky_relu2': nn.LeakyReLU(0.2),
            'attention1': AttentionLayer(128),
            'res_block1': ResidualBlock(128, 128, num_classes),
            'conv3': nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            'leaky_relu3': nn.LeakyReLU(0.2),
            'attention2': AttentionLayer(256),
            'res_block2': ResidualBlock(256, 256, num_classes),
            'conv4': nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, stride=1, padding=0)),
            'sigmoid': nn.Sigmoid()
        })

    def forward(self, img, labels):
        embedding = self.embedding(labels).view(
            labels.size(0), 1, self.img_size, self.img_size)
        img = torch.cat([img, embedding], dim=1)

        for name, module in self.model.items():
            if isinstance(module, ResidualBlock):
                img = module(img, labels)
            else:
                img = module(img)

        return img


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = ConditionalBatchNorm2d(
            out_channels, num_classes) if num_classes else nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(
            out_channels, num_classes) if num_classes else nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.ModuleDict({
                'conv': nn.Conv2d(in_channels, out_channels, 1),
                'cond_bn': ConditionalBatchNorm2d(
                    out_channels, num_classes) if num_classes else nn.BatchNorm2d(out_channels),
            })
        else:
            self.shortcut = nn.ModuleDict({
                'eye': nn.Identity()
            })

    def forward(self, x, labels):
        for name, module in self.shortcut.items():
            if isinstance(module, ConditionalBatchNorm2d):
                identity = module(identity, labels)
            else:
                identity = module(x)

        x = self.relu(self.bn1(self.conv1(x), labels))
        x = self.bn2(self.conv2(x), labels)
        x += identity
        x = self.relu(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(
            batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(
            1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, labels):
        out = self.bn(x)
        gamma, beta = self.embed(labels).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + \
            beta.view(-1, self.num_features, 1, 1)
        return out
