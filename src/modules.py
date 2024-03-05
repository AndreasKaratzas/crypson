
import torch
import numpy as np

from torch import nn
    

class SelfAttention(nn.Module):
    """Self-attention module for GANs.
    
    Parameters
    ----------
    in_dim : int
        Number of input channels.
    """

    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        # Apply softmax to the product of Q and K.
        self.softmax = nn.Softmax(dim=-2)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input feature maps (B, C, W, H).
        
        Returns
        -------
        torch.Tensor
            Attention value + input feature.
        """
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            batch_size, -1, width * height).permute(0, 2, 1)  # B x (W*H) x C
        proj_key = self.key_conv(x).view(
            batch_size, -1, width * height)  # B x C x (W*H)
        energy = torch.bmm(proj_query, proj_key)  # Transpose check
        attention = self.softmax(energy)  # B x (W*H) x (W*H)
        proj_value = self.value_conv(x).view(
            batch_size, -1, width * height)  # B x C x (W*H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)

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

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

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
