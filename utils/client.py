
import json
import torch
import socket

from copy import deepcopy
from rich import print as rprint


class Client:
    def __init__(self, host='127.0.0.1', port=8000, 
                 logger=None, verbose=False,
                 generator=None, autoencoder=None,
                 cls_indices=None, device='cpu',
                 img_size=32, z_dim=64, latent_dim=8):
        self.host = host
        self.port = port
        self.logger = logger
        self.verbose = verbose
        self.generator = generator
        self.autoencoder = autoencoder
        self.cls_indices = cls_indices
        self.device = device
        self.img_size = img_size
        self.z_dim = z_dim
        self.latent_dim = latent_dim
        self.response = None

        self._ensure_evaluation_mode()
    
    def _ensure_evaluation_mode(self):
        if self.generator:
            self.generator.eval()
        if self.autoencoder:
            self.autoencoder.eval()
        self.generator.to(self.device)
        self.autoencoder.to(self.device)
    
    def tokenize(self):
        tokens = []
        for char in self.prompt:
            tokens.append(self.cls_indices[char])
        self.tokens = tokens

    @torch.no_grad()
    def encrypt(self):
        space_index = -1  # Index of the space character
        newline_index = -2  # Index of the newline character

        batch_labels = torch.tensor(
            [class_idx for class_idx in self.cls_indices if (
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
        gen_img = self.generator(z, batch_labels).detach(
        ).cpu().view(-1, 1, self.img_size, self.img_size)
        gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())
        mu, _ = self.autoencoder.encode(gen_img)
        space_latent = torch.zeros(1, self.latent_dim)
        newline_latent = torch.ones(1, self.latent_dim) * (-1)
        merged_latents = []
        curr_count = 0
        for token in self.tokens:
            if token == space_index:
                merged_latents.append(space_latent)
            elif token == newline_index:
                merged_latents.append(newline_latent)
            else:
                merged_latents.append(mu[curr_count])
                curr_count += 1
        
        merged_latents = torch.stack(merged_latents)
        self.encoded_message = merged_latents.tolist()

    def encrypt_and_send(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))

            if self.logger:
                self.logger.info(f"Connected to {self.host}:{self.port}")

            if self.verbose and not self.logger:
                rprint(f"Connected to {self.host}:{self.port}")

            self.encrypt()
            if self.prompt == 'exit':
                sock.sendall(self.prompt.encode('utf-8'))
                return
            else:
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
        prompt = input('Enter a prompt for encryption: ')
        if not prompt.isalnum():
            self.prompt = ''.join([' ' if not c.isalnum() else c for c in prompt])
        if self.verbose:
            rprint(f'Prompt: {prompt}')
        if self.logger:
            self.logger.info(f'Prompt: {prompt}')
        self.num_chars = len([c for c in prompt if c.isalnum()])

    def run(self):
        while True:
            self.prompt_user()
            self.encrypt_and_send()
            rprint(self.response)
            if self.prompt == 'exit':
                break


if __name__ == "__main__":
    client = Client()
    client.run()
