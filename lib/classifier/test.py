
import sys
sys.path.append('../../')

import os
import torch
import warnings
import lightning.pytorch as pl

from argparse import ArgumentParser
from rich import print as rprint
from pathlib import Path

from lib.gan.modules import Generator
from lib.vae.modules import AutoEncoder
from lib.classifier.modules import Classifier
from lib.classifier.engine import Engine
from lib.classifier.dataset import GenEMNISTDataModule
from lib.classifier.registry import CustomProgressBar
from lib.classifier.utils import get_elite
from utils.emb import TimeEmbedding

def main(args):
    pl.seed_everything(args.seed)
    sys.path += [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]

    device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')

    # Load pretrained models
    # TODO: Use the `args.resume` argument with `get_elite` to load the best checkpoint
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

    classifier = Classifier(in_dim=args.latent_dim, h_channels=args.hidden_channels,
                            auto_ckpt=autoencoder, img_size=args.resolution,
                            num_classes=args.num_classes, dropout_rate=args.dropout_rate,)
    ckp = torch.load(args.classifier, map_location=device)
    classifier.load_state_dict(ckp.get('classifier'))
    classifier.eval()

    lm = Engine(classifier=classifier, lr=args.lr,)

    # DataModule
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    dm = GenEMNISTDataModule(batch_size=args.batch_size, val_split=args.val_split,
                             num_workers=args.num_workers, num_classes=args.num_classes,
                             generator=generator, train_size=args.train_size,
                             test_size=args.test_size, z_dim=args.z_dim,
                             autoencoder=autoencoder,)

    # Callbacks
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    l_callbacks = []

    progress_bar = CustomProgressBar()
    l_callbacks.append(progress_bar)

    trainer = pl.Trainer(accelerator='gpu', devices=args.gpus,
                        callbacks=l_callbacks,
                        logger=False)
    try:
        if args.enc:
            if args.proj_path is None:
                args.proj_path = Path.cwd().parent
                rprint(f'Project path not provided. Using {args.proj_path}')
            time_m_ckpt_path = Path(
                'checkpoints/time/epoch_00000-loss_0.00000.ckpt')
            rprint(
                f"Pooling time module from {args.proj_path / time_m_ckpt_path}")

            time_mod = TimeEmbedding(dim=8, num_time_embeds=1, device=device)
            ckp = torch.load(
                args.proj_path / time_m_ckpt_path, map_location=device)
            time_mod.load_state_dict(ckp.get('time'))
            time_mod.eval()
            
            return generator, autoencoder, time_mod
    except Exception as e:
        rprint(f'Error: {e}')    
    
    try:
        if args.dec:
            if args.proj_path is None:
                args.proj_path = Path.cwd().parent
                rprint(f'Project path not provided. Using {args.proj_path}')
            time_m_ckpt_path = Path(
                'checkpoints/time/epoch_00000-loss_0.00000.ckpt')
            rprint(
                f"Pooling time module from {args.proj_path / time_m_ckpt_path}")

            time_mod = TimeEmbedding(dim=8, num_time_embeds=1, device=device)
            ckp = torch.load(
                args.proj_path / time_m_ckpt_path, map_location=device)
            time_mod.load_state_dict(ckp.get('time'))
            time_mod.eval()

            return classifier, time_mod
    except Exception as e:
        rprint(f'Error: {e}')

    if args.debug:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#inference
        trainer.test(lm, dm)
    else:
        return trainer, lm, dm


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    parser = ArgumentParser()
    # program level args
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--val-split', default=0.2, type=float)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--num-classes', default=47, type=int)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--hidden-channels', nargs='+', default=[16, 32, 32, 64], type=int)
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
    parser.add_argument('--debug', action="store_true")

    # trainer level args
    args = parser.parse_args()
    main(args)
