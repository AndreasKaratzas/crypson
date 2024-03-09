
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.embedding = nn.Embedding(num_classes, latent_dim)
        self.init_size = img_size // 4
        self.fc = nn.Linear(latent_dim, 128 * self.init_size ** 2)
        self.res_block1 = ResidualBlock(128, 128, num_classes)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.res_block2 = ResidualBlockWShortcut(128, 64, num_classes)
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(64, 1, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, z, labels):
        embedding = self.embedding(labels)
        z = torch.mul(z, embedding)
        z = self.fc(z)
        z = z.view(z.size(0), 128, self.init_size, self.init_size)
        z = self.res_block1(z, labels)
        z = self.upsample1(z)
        z = self.res_block2(z, labels)
        z = self.upsample2(z)
        z = self.conv(z)
        img = self.tanh(z)
        img = img.view(img.size(0), -1)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.embedding = nn.Embedding(num_classes, img_size * img_size)
        self.conv1 = nn.utils.spectral_norm(
            nn.Conv2d(2, 64, 4, stride=2, padding=1))
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(64, 128, 4, stride=2, padding=1))
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.res_block = ResidualBlock(128, 128, num_classes)
        self.conv3 = nn.utils.spectral_norm(
            nn.Conv2d(128, 1, 4, stride=1, padding=0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, labels):
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        embedding = self.embedding(labels).view(
            labels.size(0), 1, self.img_size, self.img_size)
        z = torch.cat([img, embedding], dim=1)
        z = self.leaky_relu1(self.conv1(z))
        z = self.leaky_relu2(self.conv2(z))
        z = self.res_block(z, labels)
        z = self.conv3(z)
        z = self.sigmoid(z)
        return z


class ResidualBlockWShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=62):
        super(ResidualBlockWShortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.shortcut_bn = ConditionalBatchNorm2d(out_channels, num_classes)

    def forward(self, x, labels):
        identity = self.shortcut_bn(self.shortcut_conv(x), labels)
        x = self.relu(self.bn1(self.conv1(x), labels))
        x = self.bn2(self.conv2(x), labels)
        x += identity
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes=62):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)

    def forward(self, x, labels):
        identity = x
        x = self.relu(self.bn1(self.conv1(x), labels))
        x = self.bn2(self.conv2(x), labels)
        x += identity
        x = self.relu(x)
        return x


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes=62):
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
