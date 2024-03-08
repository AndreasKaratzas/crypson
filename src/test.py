
import sys
sys.path.append('../')

import os
import sys
import torch
import warnings
import lightning.pytorch as pl

from argparse import ArgumentParser

from src.modules import (
    Generator, Discriminator)
from src.engine import Engine
from src.dataset import EMNISTDataModule
from src.registry import CustomProgressBar
from src.utils import get_elite


def main(args):
    pl.seed_everything(args.seed)
    sys.path += [os.path.abspath(".."),
                 os.path.abspath("."), os.path.abspath("../..")]

    device = torch.device(
        f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    generator = Generator(args.z_dim, 62, args.resolution).to(device)
    discriminator = Discriminator(62, args.resolution).to(device)
    lm = Engine(generator=generator, discriminator=discriminator,
                num_classes=62, z_dim=args.z_dim, lr=args.lr, betas=args.betas,
                clip_grad_norm=args.clip_grad_norm, lnp=None, wandb_logger=None)

    # model checkpoint
    # https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#automatic-saving
    checkpoint_dirpath = os.path.join(
        args.output, 'DCGan')
    best_checkpoint = get_elite(os.listdir(checkpoint_dirpath))
    ckp = torch.load(os.path.join(checkpoint_dirpath,
                     best_checkpoint), map_location=device)
    lm.generator.load_state_dict(ckp.get('generator'))
    lm.discriminator.load_state_dict(ckp.get('discriminator'))
    lm.eval()

    # Callbacks
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    l_callbacks = []
    progress_bar = CustomProgressBar()
    l_callbacks.append(progress_bar)

    trainer = pl.Trainer(accelerator='gpu', devices=args.gpus,
                         callbacks=l_callbacks,
                         logger=False)
    
    dm = EMNISTDataModule(data_dir=args.dataset, num_workers=args.num_workers,
                          batch_size=args.batch_size, val_split=args.val_split,
                          image_size=args.resolution)

    if args.debug:
        # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#inference
        trainer.test(lm, datamodule=dm)
    else:
        return trainer, lm, dm, checkpoint_dirpath


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    parser = ArgumentParser()
    # program level args
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--cache', action="store_true")
    parser.add_argument('--output', default='train', type=str)
    parser.add_argument('--experiment', default='DCGan', type=str)
    # NOTE: Batch size is advised to be 1 for evaluation.
    #       That way we can also plot the results.
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--resolution', default=28, type=int)
    parser.add_argument('--dataset', default='../data', type=str)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--val-split', default=0.15, type=float)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--z-dim', default=100, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--betas', nargs='+', default=(0.5, 0.999), type=float)
    # trainer level args
    args = parser.parse_args()
    main(args)
