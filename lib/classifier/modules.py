
import torch
import torch.nn as nn

from typing import List


class Classifier(nn.Module):
    def __init__(self, in_dim: int, h_channels: List[int], auto_ckpt: nn.Module,
                 img_size=32, num_classes: int = 47, dropout_rate=0.2):
        super(Classifier, self).__init__()

        num_layers = len(h_channels)
        h_channels = [1] + h_channels
        h_channels = h_channels[::-1]

        features = []
        features.append(
            nn.Linear(in_dim, h_channels[0] * (img_size // (2 ** num_layers)) ** 2))
        features.append(nn.ReLU())
        features.append(nn.Unflatten(
            1, (h_channels[0], img_size // 2 ** num_layers, img_size // 2 ** num_layers)))
        for i in range(num_layers):
            features.append(nn.ConvTranspose2d(
                h_channels[i], h_channels[i+1], kernel_size=3, stride=2, padding=1, output_padding=1))
            features.append(nn.BatchNorm2d(h_channels[i+1]))
            features.append(nn.ReLU())
        features.append(
            nn.Conv2d(h_channels[-1], 1, kernel_size=3, padding=1))
        features.append(nn.Sigmoid())

        self._load_from_pretrained_model(features, auto_ckpt)
        self._n_out_features(features, img_size)

        features.append(nn.Flatten())
        features.append(nn.Linear(self.n_out_features, 512))
        features.append(nn.ReLU())
        features.append(nn.Dropout(dropout_rate))
        features.append(nn.Linear(512, num_classes))
        features.append(nn.Softmax(dim=1))
        self.features = nn.Sequential(*features)

        for i, module in enumerate(self.features[:(3 + (2 * num_layers) + 2)]):
            print(f"Freezing module {i}: {module.__class__.__name__}")
            for param in module.parameters():
                param.requires_grad = False

    def _load_from_pretrained_model(self, features, auto_ckpt):
        for i, (pretrained_layer, layer) in enumerate(zip(auto_ckpt.decode.decoder, features)):
            if isinstance(pretrained_layer, nn.Sequential):
                for j, (pretrained_sublayer, sublayer) in enumerate(zip(pretrained_layer, layer)):
                    sublayer.load_state_dict(pretrained_sublayer.state_dict())
            else:
                layer.load_state_dict(pretrained_layer.state_dict())

    def _n_out_features(self, features, img_size):
        x = torch.zeros(1, 1, img_size, img_size)
        for layer in features:
            x = layer(x)
        self.n_out_features = x.flatten().shape[0]

    def forward(self, latents):
        return self.features(latents)
