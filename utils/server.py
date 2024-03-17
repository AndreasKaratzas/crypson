
import os
import json
import time
import torch
import psutil
import shutil
import socket
import argparse
import threading


class Server:
    def __init__(self, host='127.0.0.1', port=8000, 
                 decoder=None, debug=False):
        self.host = host
        self.port = port
        self.debug = debug
        self.decoder = decoder

    def send_large_data(self, conn, data):
        data_bytes = data.encode('utf-8')
        chunk_size = 1024
        for i in range(0, len(data_bytes), chunk_size):
            conn.sendall(data_bytes[i:i+chunk_size])
        return

    @torch.no_grad()
    def handle_client(self, conn, addr):
        print(f"Connected to {addr}")

        message = conn.recv(1024).decode('utf-8')
        if not message:
            conn.sendall(json.dumps({'message_received': {}}).encode('utf-8'))
            print(f"Disconnected from {addr}")
            conn.close()

        if message == 'exit':
            conn.sendall(json.dumps({'message_received': {}}).encode('utf-8'))
            print(f"Disconnected from {addr}")
            conn.close()

        print(f"Received encrypted message: {message}")

        

        self.send_large_data(conn, json.dumps(
            {'message_received': decrpyted_message}))

        print(f"Disconnected from {addr}")
        conn.close()

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"Server is listening on {self.host}:{self.port}")

            while True:
                conn, addr = s.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.start()
                thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='IP address of the server')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port of the server')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug messages')
    args = parser.parse_args()

    server = Server(
        host=args.host, port=args.port, debug=args.debug,)
    server.run()
