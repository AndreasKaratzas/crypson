
import sys
sys.path.append('./')

import os
import json
import torch
import argparse
import warnings

from lib.vae.modules import AutoEncoder
from lib.gan.modules import Generator
from utils.logger import Logger
from utils.client import Client
from utils.server import Server


def main(args):
    if not args.en_server_mode and not args.en_client_mode:
        raise ValueError('Either server or client mode should be enabled.')

    device = torch.device(
        f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')

    # Load pretrained models
    autoencoder = AutoEncoder(in_channels=1, hidden_channels=args.hidden_channels,
                              latent_dim=args.latent_dim, img_size=args.resolution,)
    ckp = torch.load(args.autoencoder, map_location=device)
    autoencoder.load_state_dict(ckp.get('vae'))
    autoencoder.eval()

    generator = Generator(latent_dim=args.z_dim, img_size=args.resolution,
                          num_classes=args.num_classes)
    ckp = torch.load(args.generator, map_location=device)
    generator.load_state_dict(ckp.get('generator'))
    generator.eval()

    # initialize a logger
    os.makedirs('./data/logs', exist_ok=True)
    lnp = Logger(logger_name='server' if args.en_server_mode else 'client', 
                 verbose=args.verbose, filepath='./data/logs')
    
    with open(args.classes_path, 'r') as file:
        class_to_idx = json.load(file)

    class_to_idx[' '] = -1
    class_to_idx['\n'] = -2

    if args.en_client_mode:
        client = Client(host=args.ip, port=args.port, logger=lnp, verbose=args.verbose)
        client.run()
    else:
        server = Server()
        server.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate images using a trained model.')
    parser.add_argument('--generator', type=str, required=True,
                        help='Path to the generator model checkpoint.')
    parser.add_argument('--autoencoder', type=str, required=True,
                        help='Path to the autoencoder model checkpoint.')
    parser.add_argument('--classes-path', type=str, default='./data/idx_to_class.json',
                        help='Path to the classes JSON path.')
    parser.add_argument('--en-server-mode', action='store_true',
                        help='Enable server mode.')
    parser.add_argument('--en-client-mode', action='store_true',
                        help='Enable client mode.')
    parser.add_argument('--ip', type=str, default='0.0.0.0', 
                        help='IP address')
    parser.add_argument('--port', type=int, default=8080, 
                        help='Port number')  
    parser.add_argument('--latent-dim', type=int, default=64,
                        help='Dimension of the latent space.')
    parser.add_argument('--num-classes', type=int, default=47,
                        help='Number of classes in the dataset.')
    parser.add_argument('--img-size', type=int, default=32,
                        help='Size of the input images.')
    parser.add_argument('--z-dim', default=64, type=int,
                        help='Dimension of the latent space.')
    parser.add_argument('--hidden-channels', nargs='+', default=[16, 32, 32, 64], type=int,
                        help='Number of hidden channels in the encoder and decoder.')
    parser.add_argument('--kl-w', default=0.5, type=float,
                        help='Weight of the KL divergence loss.')
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
