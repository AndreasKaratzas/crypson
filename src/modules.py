
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn


class ConvSelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(ConvSelfAttention, self).__init__()
        # Query, key, value transformations
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        # Output softmax scaling factor (to prevent large values in softmax)
        self.scale = (in_dim // 8) ** -0.5

        # Convolution for the output of the attention to project back to the input dimensions
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(
            batch_size, -1, width * height).permute(0, 2, 1)  # B X N X C
        key = self.key_conv(x).view(
            batch_size, -1, width * height)  # B X C X N
        value = self.value_conv(x).view(
            batch_size, -1, width * height)  # B X C X N

        attention = self.softmax(torch.bmm(query, key)
                                 * self.scale)  # B X N X N
        out = torch.bmm(value, attention.permute(0, 2, 1)
                        ).view(batch_size, C, width, height)

        # Apply a residual connection
        out = self.gamma * out + x
        return out


class Generator(nn.Module):
    """Generator for a Deep Convolutional Conditional GAN.

    Parameters
    ----------
    z_dim : int
        Dimension of the latent noise vector.
    img_shape : tuple
        Shape of the input images.
    n_classes : int
        Number of classes for conditional generation.
    """

    def __init__(self, z_dim, img_shape, n_classes):
        super(Generator, self).__init__()

        self.z_dim = z_dim
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            *self.block(z_dim + n_classes, 128, normalize=False),
            *self.block(128, 256),
            *self.block(256, 512),
            *self.block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def block(self, in_feat, out_feat, normalize=True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    """Discriminator for a Deep Convolutional Conditional GAN.

    Parameters
    ----------
    img_shape : tuple
        Shape of the input images.
    n_classes : int
        Number of classes for conditional discrimination.
    """

    def __init__(self, img_shape, n_classes):
        super(Discriminator, self).__init__()

        self.img_shape = img_shape
        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1),
                         self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


class ConvGenerator(nn.Module):
    def __init__(self, z_dim, img_shape, n_classes):
        super(ConvGenerator, self).__init__()
        self.img_shape = img_shape
        self.label_emb = nn.Embedding(n_classes, np.prod(img_shape))

        # Initial size before upscaling
        self.init_size = self.img_shape[1] // 4
        self.l1 = nn.Sequential(
            nn.Linear(z_dim + np.prod(img_shape), 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ConvSelfAttention(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            ConvSelfAttention(64),
            nn.Conv2d(64, self.img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ConvDiscriminator(nn.Module):
    def __init__(self, img_shape, n_classes):
        super(ConvDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(n_classes, np.prod(img_shape))

        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0] + 1, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            ConvSelfAttention(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            ConvSelfAttention(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=0),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.25),
            ConvSelfAttention(128),
            nn.Flatten(),
            nn.Linear(128 * ((img_shape[1] // 2 ** 3) ** 2), 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_img = self.label_embedding(labels).view(
            img.size(0), 1, img.size(2), img.size(3))
        d_in = torch.cat((img, label_img), 1)  # Concatenate image and label
        validity = self.model(d_in)
        return validity


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=8):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, cls):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(cls)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=8):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, cls):
        x = self.up(x)
        x = self.conv(x)
        emb = self.emb_layer(cls)[:, :, None, None].repeat(
            1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNetGenerator(nn.Module):
    def __init__(self, z_dim=8, img_shape=(1, 28, 28), n_classes=62):
        super().__init__()
        self.label_embedding = nn.Embedding(n_classes, z_dim)

        self.inc = DoubleConv(in_channels=img_shape[0], out_channels=2)
        self.up = Up(in_channels=2, out_channels=4, emb_dim=z_dim)
        self.sa1 = SelfAttention(channels=4)
        self.down = Down(in_channels=4, out_channels=8, emb_dim=z_dim)
        self.sa2 = SelfAttention(channels=8)
        self.outc = nn.Conv2d(in_channels=8, out_channels=img_shape[0], kernel_size=1)
        self.tanh = nn.Tanh()
    
    def forward(self, noise, labels):
        l_emb = self.label_embedding(labels)
        z = self.inc(noise)
        z = self.up(z, l_emb)
        z = self.sa1(z)
        z = self.down(z, l_emb)
        z = self.sa2(z)
        z = self.outc(z)
        y_hat = self.tanh(z)
        return y_hat


class UNetDiscriminator(nn.Module):
    def __init__(self, z_dim=8, img_shape=(1, 28, 28), n_classes=62):
        super().__init__()
        self.label_embedding = nn.Embedding(n_classes, z_dim)
        h_dim = img_shape[1] // 2 ** 2

        self.inc = DoubleConv(in_channels=img_shape[0], out_channels=2)
        self.down1 = Down(in_channels=2, out_channels=4, emb_dim=z_dim)
        self.sa1 = SelfAttention(channels=4)
        self.down2 = Down(in_channels=4, out_channels=8, emb_dim=z_dim)
        self.sa2 = SelfAttention(channels=8)
        self.outc = nn.Linear(8 * h_dim * h_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, labels):
        l_emb = self.label_embedding(labels)
        z = self.inc(img)
        z = self.down1(z, l_emb)
        z = self.sa1(z)
        z = self.down2(z, l_emb)
        z = self.sa2(z)
        z = z.reshape(z.size(0), -1)
        y_hat = self.sigmoid(self.outc(z))
        return y_hat


if __name__ == "__main__":
    generator = UNetGenerator(z_dim=8, img_shape=(1, 28, 28), n_classes=62)
    discriminator = UNetDiscriminator(z_dim=8, img_shape=(1, 28, 28), n_classes=62)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator.to(device)
    discriminator.to(device)

    x = torch.randn(16, 1, 28, 28).to(device)
    y = torch.randint(0, 62, (16,)).to(device)

    print(generator(x, y).shape)
    print(discriminator(x, y))
