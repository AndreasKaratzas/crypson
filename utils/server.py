import sys
sys.path.append('../')

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import json
import time
import torch
import psutil
import shutil
import socket
import warnings
import argparse
import threading

from pathlib import Path
from rich import print as rprint

from lib.classifier.test import main as cls_main


MAX_COUNTER_VALUE = 1000000000

class Server:
    def __init__(self, host='127.0.0.1', port=8000, 
                 decoder=None, time_emb=None,
                 device='cpu', debug=False,
                 img_size=32, z_dim=64, latent_dim=8,
                 idx_to_class=None, logger=None):
        self.host = host
        self.port = port
        self.debug = debug
        self.decoder = decoder
        self.device = device
        self.time_emb = time_emb
        self.img_size = img_size
        self.z_dim = z_dim
        self.logger = logger
        self.latent_dim = latent_dim
        self.idx_to_class = idx_to_class
        self.counter = 0
        self.en_listen = True

        self._ensure_evaluation_mode()

    def _ensure_evaluation_mode(self):
        if self.decoder:
            self.decoder.eval()
        if self.time_emb:
            self.time_emb.eval()
        self.decoder.to(self.device)
        self.time_emb.to(self.device)

    def send_large_data(self, conn, data):
        data_bytes = data.encode('utf-8')
        chunk_size = 1024
        for i in range(0, len(data_bytes), chunk_size):
            conn.sendall(data_bytes[i:i+chunk_size])
        return

    @torch.no_grad()
    def handle_client(self, conn, addr, s):
        rprint(f"Connected to {addr}")
        message = bytearray()
        try:
            while True:
                chunk = conn.recv(1024)
                if not chunk:
                    break       # Connection closed by the client
                if self.debug:
                    rprint(f"Received chunk: {chunk}")
                
                message.extend(chunk)

                try:
                    complete_message = json.loads(message.decode('utf-8'))
                    if self.debug:
                        rprint(f"Received complete message: {complete_message}")
                    if self.logger:
                        self.logger.info(f"Received complete message: {complete_message}")
                    self.tokens = complete_message['message']
                    space_latent = torch.zeros(
                        1, self.latent_dim).to(self.device)
                    newline_latent = torch.ones(1, self.latent_dim).to(self.device) * (-1)
                    t = torch.arange(len(self.tokens), device=self.device).float() + self.counter
                    t_hat = self.time_emb(t)
                    curr_count = 0
                    merged_chars = []
                    for idx, token in enumerate(self.tokens):
                        latent = torch.tensor(token, dtype=torch.float).to(
                            self.device).view(1, -1) - t_hat[idx].view(1, -1)
                        if torch.all(latent == space_latent):
                            merged_chars.append(" ")
                        elif torch.all(latent == newline_latent):
                            merged_chars.append("\n")
                        else:
                            pred_char = self.decoder(
                                latent).detach().cpu()
                            pred_char = torch.argmax(pred_char, dim=1).item()
                            pred_char = self.idx_to_class[pred_char]
                            merged_chars.append(pred_char)
                            curr_count += 1

                    rprint(f"Decoded message: {merged_chars}")
                    if self.logger:
                        self.logger.info(f"Decoded message: {merged_chars}")
                    self.merged_chars = merged_chars
                    self.counter += len(self.tokens)
                    if self.counter > MAX_COUNTER_VALUE:
                        self.counter = 0
                    break       # Exit the loop after processing the complete message
                except json.JSONDecodeError:
                    continue    # JSON not yet complete, continue receiving data

        except socket.timeout:
            rprint("Connection timed out due to inactivity.")
        finally:
            if not hasattr(self, 'merged_chars'):
                rprint(f"Exiting after receiving `{message}` command.")
                self.en_listen = False
                self.merged_chars = 'Shutting down ...'
                if self.logger:
                    self.logger.info("Shutting down ...")
            
            conn.sendall(json.dumps(
                {'message_received': self.merged_chars}).encode('utf-8'))
            del self.merged_chars
            rprint(f"Disconnected from {addr}")
            conn.close()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((self.host, self.port))
                s.listen(1)
                rprint(f"Server is listening on {self.host}:{self.port}")

                while self.en_listen:
                    conn, addr = s.accept()
                    thread = threading.Thread(target=self.handle_client, args=(conn, addr, s))
                    thread.start()
                    thread.join()
            finally:
                s.close()


"""Example usage:

>>> python server.py --generator "../checkpoints/gan/epoch_00199-loss_0.63360.ckpt" --autoencoder "../checkpoints/vae/epoch_00098-loss_7669.00684.ckpt" --classifier "../checkpoints/classifier/epoch_00099-loss_0.90071.ckpt" --proj-path "../" --dec
"""
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0',
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

    if not args.dec:
        args.enc = False
        args.dec = True
    classifier, time_mod = cls_main(args)

    cls_idx_path = args.proj_path / Path('data/idx_to_class.json')
    with open(cls_idx_path, 'r') as f:
        cls_indices = json.load(f)
    idx_to_class = {v: k for k, v in cls_indices.items()}

    server = Server(
        host=args.host, port=args.port, debug=args.debug,
        decoder=classifier, time_emb=time_mod,
        device='cuda' if torch.cuda.is_available() else 'cpu', 
        img_size=args.resolution, z_dim=args.z_dim, latent_dim=args.latent_dim,
        idx_to_class=idx_to_class)
    server.run()
