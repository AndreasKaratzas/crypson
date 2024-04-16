
import sys
sys.path.append('./')

import os
import json
import torch
import argparse
import warnings

from pathlib import Path

from lib.classifier.test import main as cls_main
from utils.logger import Logger
from utils.client import Client
from utils.server import Server


def main(args):
    if not args.en_server_mode and not args.en_client_mode:
        raise ValueError('Either server or client mode should be enabled.')

    device = torch.device(
        f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')

    # initialize a logger
    os.makedirs('./data/logs', exist_ok=True)
    lnp = Logger(logger_name='server' if args.en_server_mode else 'client', 
                 verbose=args.verbose, filepath='./data/logs')
    
    with open(Path(args.classes_path), 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_to_idx[' '] = -1
    class_to_idx['\n'] = -2

    if args.en_client_mode:
        args.enc = True
        args.dec = False
        generator, autoencoder, time_mod = cls_main(args)
        client = Client(host=args.ip, port=args.port, debug=args.debug, generator=generator, 
                        autoencoder=autoencoder, time_emb=time_mod, logger=lnp, device=device,
                        img_size=args.resolution, z_dim=args.z_dim, latent_dim=args.latent_dim,
                        verbose=args.verbose, cls_indices=class_to_idx)
        client.run()
    else:
        args.enc = False
        args.dec = True
        classifier, time_mod = cls_main(args)
        server = Server(host=args.ip, port=args.port, debug=args.debug, decoder=classifier, 
                        time_emb=time_mod, device=device, img_size=args.resolution, z_dim=args.z_dim, 
                        latent_dim=args.latent_dim, idx_to_class=idx_to_class, logger=lnp)
        server.run()

"""Example usage:

    * For server mode:
    >>> python main.py --generator "./checkpoints/gan/epoch_00199-loss_0.63360.ckpt" --autoencoder "./checkpoints/vae/epoch_00098-loss_7669.00684.ckpt" --classifier "./checkpoints/classifier/epoch_00099-loss_0.90071.ckpt" --en-server-mode

    * For client mode:
    >>> python main.py --generator "./checkpoints/gan/epoch_00199-loss_0.63360.ckpt" --autoencoder "./checkpoints/vae/epoch_00098-loss_7669.00684.ckpt" --classifier "./checkpoints/classifier/epoch_00099-loss_0.90071.ckpt" --en-client-mode --ip "127.0.0.1"
"""
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='Generate images using a trained model.')
    parser.add_argument('--ip', type=str, default='0.0.0.0',
                        help='IP address of the server')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port of the server')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug messages')
    parser.add_argument('--en-server-mode', action='store_true',
                        help='Enable server mode.')
    parser.add_argument('--en-client-mode', action='store_true',
                        help='Enable client mode.')
    parser.add_argument('--enc', action="store_true", default=False,
                        help='Enable encoder mode.')
    parser.add_argument('--dec', action="store_true", default=False,
                        help='Enable decoder mode.')
    # classifier args
    parser.add_argument('--seed', default=33, type=int,
                        help='Random seed for reproducibility.')
    parser.add_argument('--generator', type=str, required=True,
                        help='Path to the generator model checkpoint.')
    parser.add_argument('--autoencoder', type=str, required=True,
                        help='Path to the autoencoder model checkpoint.')
    parser.add_argument('--classifier', type=str, required=True,
                        help='Path to the classifier model checkpoint.')
    parser.add_argument('--classes-path', type=str, default='./data/idx_to_class.json',
                        help='Path to the classes JSON path.')
    parser.add_argument('--proj-path', type=str, default='./',
                        help='Path to the project directory.')
    parser.add_argument('--latent-dim', type=int, default=8,
                        help='Dimension of the latent space.')
    parser.add_argument('--num-classes', type=int, default=47,
                        help='Number of classes in the dataset.')
    parser.add_argument('--resolution', type=int, default=32,
                        help='Size of the input images.')
    parser.add_argument('--z-dim', default=64, type=int,
                        help='Dimension of the latent space.')
    parser.add_argument('--hidden-channels', nargs='+', default=[32, 64, 128, 256], type=int,
                        help='Number of hidden channels in the encoder and decoder.')
    parser.add_argument('--kl-w', default=0.5, type=float,
                        help='Weight of the KL divergence loss.')
    parser.add_argument('--dropout-rate', default=0.2, type=float,
                        help='Dropout rate for the classifier.')
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='Learning rate.')
    parser.add_argument('--batch-size', default=1024, type=int,
                        help='Batch size.')
    parser.add_argument('--num-workers', default=8, type=int,
                        help='Number of workers for the dataloader.')
    parser.add_argument('--val-split', default=0.2, type=float,
                        help='Validation split.')
    parser.add_argument('--train-size', default=0, type=int,
                        help='Size of the training dataset.')
    parser.add_argument('--test-size', default=0, type=int,
                        help='Size of the test dataset.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose mode.')
    parser.add_argument('--gpus', nargs='+', default=[0], type=int,
                        help='List of GPUs to use.')
    args = parser.parse_args()

    main(args)
