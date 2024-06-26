
import sys
sys.path.append('../../')

import os
import torch
import warnings
import lightning.pytorch as pl

from argparse import ArgumentParser

from lib.gan.modules import Generator
from lib.vae.engine import Engine
from lib.vae.modules import AutoEncoder
from lib.vae.dataset import GenEMNISTDataModule
from lib.vae.registry import CustomProgressBar
from lib.vae.utils import get_elite


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

    ckp = torch.load(args.generator, map_location=device)
    lm = Engine(vae=autoencoder, lr=args.lr,
                kl_w=args.kl_w, img_size=args.resolution)

    # TODO: Use the `args.resume` argument with `get_elite` to load the best checkpoint
    generator = Generator(latent_dim=args.z_dim,
                          img_size=args.resolution,
                          num_classes=args.num_classes)
    generator.load_state_dict(ckp.get('generator'))
    generator.eval()

    # DataModule
    # https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    dm = GenEMNISTDataModule(batch_size=args.batch_size, val_split=args.val_split, 
                             num_workers=args.num_workers, num_classes=args.num_classes, 
                             generator=generator, train_size=args.train_size, 
                             test_size=args.test_size, z_dim=args.z_dim,)

    # Callbacks
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    l_callbacks = []

    progress_bar = CustomProgressBar()
    l_callbacks.append(progress_bar)

    trainer = pl.Trainer(accelerator='gpu', devices=args.gpus,
                        callbacks=l_callbacks,
                        logger=False)

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
    parser.add_argument('--kl-w', default=0.5, type=float)
    parser.add_argument('--generator', type=str)
    parser.add_argument('--autoencoder', type=str)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--z-dim', default=64, type=int)
    parser.add_argument('--train-size', default=100, type=int)
    parser.add_argument('--test-size', default=470, type=int)
    parser.add_argument('--resolution', default=32, type=int)
    parser.add_argument('--alias', type=str, default=None)
    parser.add_argument('--debug', action="store_true")

    # trainer level args
    args = parser.parse_args()
    main(args)
