
import sys
sys.path.append('../../')

import os
import sys
import time
import wandb
import torch
import warnings
import lightning.pytorch as pl

from rich import print as rprint
from rich.syntax import Syntax
from argparse import ArgumentParser

from lib.gan.logger import Logger
from lib.gan.engine import Engine
from lib.gan.modules import (
    Generator, Discriminator)
from lib.gan.dataset import EMNISTDataModule
from lib.gan.registry import CustomProgressBar
from lib.gan.info import collect_env_details
from lib.gan.utils import get_elite


def main(args):
    ts_script = time.time()
    pl.seed_everything(args.seed)

    # create dirs for saving
    os.makedirs(os.path.join(args.output, 'log'), exist_ok=True)

    message = """
    To start a tensorboard instance, run the following command:
        >>> tensorboard --logdir=./experiments/ --host localhost --port 8888
    """
    syntax = Syntax(message, "python", theme="monokai", line_numbers=False)
    rprint(syntax)

    if len(args.gpus) > 1:
        print(f'Warning: Multiple GPUs are not supported yet. Using GPU {args.gpus[0]}')
        args.gpus = args.gpus[0]

    # logger
    # https://pytorch-lightning.readthedocs.io/en/latest/common/loggers.html#tensorboard
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html
    # loggers need info from args, so have to run args first before loggers
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(args.output, 'log'),
                                             name=args.experiment,
                                             log_graph=False)
    '''
    The tensorboard is creating a new version unless we fix it with a new version name.
    '''
    wandb_logger = pl.loggers.WandbLogger(save_dir=os.path.join(args.output, 'log'),
                                          offline=False,  # cannot log model while offline
                                          project=args.experiment,
                                          name=args.alias if args.alias else None,
                                          resume=False)
    lnp = Logger(tb_logger, wandb_logger, args.experiment, args.output)
    lnp.lnp('Loggers start')
    lnp.lnp('ts_script: ' + str(ts_script))

    sys.path += [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]
    lnp.lnp(collect_env_details())

    strargs = ''
    for (k,v) in vars(args).items():
        strargs += str(k) + ': ' + str(v) + '\n'
    lnp.lnp('ARGUMENTS:\n' + strargs)

    # ------------
    # LightningModule
    # https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    # ------------
    lnp.lnp('MAIN LightningModule')
    generator = Generator(args.z_dim, args.resolution, args.num_classes)
    discriminator = Discriminator(args.resolution, args.num_classes)
    lm = Engine(generator=generator, discriminator=discriminator, 
                num_classes=args.num_classes, z_dim=args.z_dim, lr=args.lr, betas=args.betas,
                lnp=lnp, wandb_logger=wandb_logger)
    for n,p in lm.named_parameters():
        lnp.lnp(n + ': ' + str(p.data.shape))

    # Callbacks
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    lnp.lnp('MAIN callbacks')
    l_callbacks = []

    # model checkpoint
    # https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#automatic-saving
    checkpoint_dirpath = os.path.join(args.output, 'DCGan')
    progress_bar = CustomProgressBar()
    l_callbacks.append(progress_bar)

    if args.resume or args.ckpt_path:
        device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
        if args.ckpt_path is None:
            best_checkpoint = get_elite(os.listdir(checkpoint_dirpath))
            if os.path.exists(os.path.join(checkpoint_dirpath, best_checkpoint)):
                ckp = torch.load(os.path.join(checkpoint_dirpath, best_checkpoint), map_location=device)
                lm.generator.load_state_dict(ckp.get('generator'))
                lm.discriminator.load_state_dict(ckp.get('discriminator'))
            else:
                print(f'No checkpoint found at {os.path.join(checkpoint_dirpath, best_checkpoint)}')
        else:
            ckp = torch.load(args.ckpt_path, map_location=device)
            lm.generator.load_state_dict(ckp.get('generator'))
            lm.discriminator.load_state_dict(ckp.get('discriminator'))

    cbModelCheckpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=10,
        monitor="d_loss",
        mode="min",
        dirpath=checkpoint_dirpath,
        filename="epoch_{epoch:05d}-loss_{d_loss:.5f}",
        auto_insert_metric_name=False,
        save_last=True,)
    l_callbacks.append(cbModelCheckpoint)
    
    lnp.lnp('MAIN trainer')
    trainer = pl.Trainer(max_epochs=args.num_epochs,
                         accelerator='gpu', devices=args.gpus,
                         logger=[tb_logger, wandb_logger],
                         callbacks=l_callbacks,
                         enable_checkpointing=True,
                         num_sanity_val_steps=0,)
    dm = EMNISTDataModule(data_dir=args.dataset, num_workers=args.num_workers,
                          batch_size=args.batch_size, val_split=args.val_split,
                          image_size=args.resolution)

    # fit
    lnp.lnp('MAIN fit')
    trainer.fit(lm, datamodule=dm)

    # exit wandb -- https://github.com/Lightning-AI/lightning/issues/5212
    wandb.finish()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    # https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html
    parser = ArgumentParser()
    # program level args
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--output', default='train', type=str)
    parser.add_argument('--experiment', default='DCGan', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--num-classes', default=47, type=int)
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--val-split', default=0.05, type=float)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--z-dim', default=128, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--betas', nargs='+', default=[0.5, 0.999], type=float)
    parser.add_argument('--resolution', default=32, type=int)
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--dataset', default='../../data', type=str)
    parser.add_argument('--alias', type=str, default=None)

    # trainer level args
    args = parser.parse_args()
    main(args)
