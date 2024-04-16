
import sys
sys.path.append('../')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import time
import json
import torch
import socket
import warnings
import argparse
import ujson as json

from copy import deepcopy
from pathlib import Path
from rich import print as rprint

from lib.classifier.test import main as cls_main
from utils.logger import Logger

MAX_COUNTER_VALUE = 1000000000

class Client:
    def __init__(self, host='127.0.0.1', port=8080, 
                 logger=None, verbose=False, time_emb=None,
                 generator=None, autoencoder=None,
                 cls_indices=None, device='cpu',
                 img_size=32, z_dim=64, latent_dim=8,
                 debug=False):
        self.host = host
        self.port = port
        self.logger = logger
        self.verbose = verbose
        self.time_emb = time_emb
        self.generator = generator
        self.autoencoder = autoencoder
        self.cls_indices = cls_indices
        self.device = device
        self.img_size = img_size
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.response = None
        self.counter = 0

        self._ensure_evaluation_mode()
    
    def _ensure_evaluation_mode(self):
        if self.generator:
            self.generator.eval()
        if self.autoencoder:
            self.autoencoder.eval()
        if self.time_emb:
            self.time_emb.eval()
        self.generator.to(self.device)
        self.autoencoder.to(self.device)
        self.time_emb.to(self.device)
    
    def tokenize(self):
        tokens = []
        for char in self.prompt:
            tokens.append(self.cls_indices[char])
        self.tokens = tokens

    @torch.no_grad()
    def encrypt(self):
        space_index = -1  # Index of the space character
        newline_index = -2  # Index of the newline character

        self.tokenize()
        batch_labels = torch.tensor(
            [class_idx for class_idx in self.tokens if (
                class_idx != space_index and class_idx != newline_index)]).to(self.device)
        if self.verbose:
            rprint(f'Batch labels: {batch_labels}')
        if self.logger:
            self.logger.info(f'Batch labels: {batch_labels}')
        z = torch.randn(len(batch_labels), self.z_dim).to(self.device)
        if self.verbose:
            rprint(f"Input noise shape: {z.shape}\n Batch labels shape: {batch_labels.shape}")
        if self.logger:
            self.logger.info(f"Input noise shape: {z.shape}")
            self.logger.info(f"Batch labels shape: {batch_labels.shape}")
        gen_img = self.generator(z, batch_labels).view(-1, 1, self.img_size, self.img_size)
        gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())
        mu, _ = self.autoencoder.encode(gen_img)
        space_latent = torch.zeros(1, self.latent_dim).to(self.device)
        newline_latent = torch.ones(1, self.latent_dim).to(self.device) * (-1)
        merged_latents = []
        t = torch.arange(len(self.tokens), device=self.device).float() + self.counter
        t_hat = self.time_emb(t)
        curr_count = 0
        for idx, token in enumerate(self.tokens):
            if token == space_index:
                merged_latents.append(space_latent + t_hat[idx].view(1, -1))
            elif token == newline_index:
                merged_latents.append(newline_latent + t_hat[idx].view(1, -1))
            else:
                merged_latents.append(mu[curr_count] + t_hat[idx].view(1, -1))
                curr_count += 1
        
        # TODO: Resolve overflow case
        self.counter += len(self.tokens)
        if self.counter > MAX_COUNTER_VALUE:
            self.counter = 0

        merged_latents = torch.stack(merged_latents).squeeze().detach().cpu()
        self.encoded_message = json.dumps({"message": merged_latents.tolist()})

    def encrypt_and_send(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))

            if self.logger:
                self.logger.info(f"Connected to {self.host}:{self.port}")

            if self.verbose and not self.logger:
                rprint(f"Connected to {self.host}:{self.port}")

            if self.prompt == 'exit':
                self.response = {}
                sock.sendall(self.prompt.encode('utf-8'))
                return
            else:
                self.encrypt()
                sock.sendall(self.encoded_message.encode('utf-8'))

            response_tmp = ""
            while True:
                chunk = sock.recv(1024).decode('utf-8')
                if not chunk:
                    break  # Connection closed
                response_tmp += chunk
                try:
                    self.response = json.loads(response_tmp)
                    break  # Received complete JSON object
                except json.decoder.JSONDecodeError:
                    pass  # JSON object not yet complete

            if not response_tmp:
                self.response = {}

    def prompt_user(self):
        """Prompt user to enter a prompt to sent.
        """
        self.prompt = input('Enter a prompt for encryption: ')
        if not self.prompt.isalnum():
            self.prompt = ''.join(
                [' ' if not c.isalnum() else c for c in self.prompt])
        if self.verbose:
            rprint(f'Prompt: {self.prompt}')
        if self.logger:
            self.logger.info(f'Prompt: {self.prompt}')
        self.num_chars = len([c for c in self.prompt if c.isalnum()])

    def run(self):
        while True:
            self.prompt_user()
            self.encrypt_and_send()
            rprint(self.response)
            if self.prompt == 'exit':
                break


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='IP address of the server')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port of the server')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug messages')

    # classifier args
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--val-split', default=0.2, type=float)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--num-classes', default=47, type=int)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--hidden-channels', nargs='+',
                        default=[32, 64, 128, 256], type=int)
    parser.add_argument('--latent-dim', default=8, type=int)
    parser.add_argument('--generator', type=str)
    parser.add_argument('--autoencoder', type=str)
    parser.add_argument('--classifier', type=str)
    parser.add_argument('--proj-path', type=str)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--enc', action="store_true")
    parser.add_argument('--dec', action="store_true")
    parser.add_argument('--dropout-rate', default=0.2, type=float)
    parser.add_argument('--z-dim', default=64, type=int)
    parser.add_argument('--train-size', default=100, type=int)
    parser.add_argument('--test-size', default=470, type=int)
    parser.add_argument('--resolution', default=32, type=int)
    parser.add_argument('--alias', type=str, default=None)

    # trainer level args
    args = parser.parse_args()

    if not args.enc:
        args.enc = True
        args.dec = False
    generator, autoencoder, time_mod = cls_main(args)

    cls_idx_path = args.proj_path / Path('data/idx_to_class.json')
    with open(cls_idx_path, 'r') as f:
        cls_indices = json.load(f)
    cls_indices[' '] = -1
    cls_indices['\n'] = -2

    logger = Logger('client_logger', 'INFO', verbose=True, filepath='client.log')
    client = Client(host=args.host, port=args.port, debug=args.debug,
                    generator=generator, autoencoder=autoencoder, time_emb=time_mod,
                    logger=logger, device='cuda' if torch.cuda.is_available() else 'cpu',
                    img_size=args.resolution, z_dim=args.z_dim, latent_dim=args.latent_dim,
                    verbose=True if args.debug else False, cls_indices=cls_indices)
    client.run()
