
import os
import wandb
# Set env variables
os.environ['WANDB_SILENT'] = "true"
os.environ["WANDB_CONSOLE"] = "wrap"

import sys
sys.path.append('../')

import torch 

from lib.vae.modules import AutoEncoder


if __name__ == "__main__":
    wandb.init()
    ckp = torch.load('../checkpoints/vae/epoch_00098-loss_7669.00684.ckpt', map_location='cpu')
    print(ckp.keys())
    ckp['vae'] = ckp.pop('dnn')
    torch.save(ckp, '../checkpoints/vae/remapped-epoch_00098-loss_7669.00684.ckpt')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = AutoEncoder(in_channels=1, hidden_channels=[32, 64, 128, 256],
                              latent_dim=8, img_size=32,)
    ckp = torch.load(
        '../checkpoints/vae/remapped-epoch_00098-loss_7669.00684.ckpt', map_location=device)
    autoencoder.load_state_dict(ckp.get('vae'))
    autoencoder.eval()

    x = torch.randn(4, 1, 32, 32)
    x_hat, mu, logvar = autoencoder(x)
    print(x_hat.shape, mu.shape, logvar.shape)
