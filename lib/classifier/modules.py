
import torch
import torch.nn as nn

from typing import List


class Classifier(nn.Module):
    def __init__(self, in_dim: int, h_channels: List[int], f_path: str, 
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

        self._load_from_pretrained_model(features, f_path)

        features.append(nn.Flatten())
        features.append(nn.Linear(h_channels[-1], 512))
        features.append(nn.ReLU())
        features.append(nn.Dropout(dropout_rate))
        features.append(nn.Linear(512, num_classes))
        features.append(nn.Softmax(dim=1))
        self.features = nn.Sequential(*features)

        for param in self.features[:(3 + (2 * num_layers) + 2)].parameters():
            print(f'param: {param}')
            param.requires_grad = False

    def _load_from_pretrained_model(self, features, f_path):
        ckp = torch.load(f_path, map_location='cpu')
        pretrained = ckp.get('vae')
        for i, layer in enumerate(pretrained):
            features[i].load_state_dict(layer.state_dict())

    def forward(self, latents):
        return self.features(latents)
