
import sys
sys.path.append('../../')

import os
import sys
import time
import wandb
import torch
import warnings
import lightning.pytorch as pl

from diffusers import AutoencoderKL
from rich import print as rprint
from rich.syntax import Syntax
from argparse import ArgumentParser

from lib.gan.modules import Generator
from lib.vqvae.logger import Logger
from lib.vqvae.engine import Engine
from lib.vqvae.modules import VAE
from lib.vqvae.dataset import GenEMNISTDataModule
from lib.vqvae.registry import CustomProgressBar
from lib.vqvae.info import collect_env_details
from lib.vqvae.utils import get_elite


def main(args):
    ts_script = time.time()
    pl.seed_everything(args.seed)

    # create dirs for saving
    os.makedirs(os.path.join(args.output, 'log'), exist_ok=True)
    device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')

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
    generator = Generator(latent_dim=args.z_dim, 
                          img_size=args.resolution, 
                          num_classes=args.num_classes)
    ckp = torch.load(args.generator, map_location=device)
    generator.load_state_dict(ckp.get('generator'))
    generator.eval()
    autoencoder = VAE(in_channels=1, hidden_channels=args.hidden_channels, 
                      num_layers=args.num_layers, latent_dim=args.latent_dim, 
                      img_size=args.resolution,)
    autoencoder = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D",
                          "DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D",
                        "UpDecoderBlock2D", "UpDecoderBlock2D"],
        block_out_channels=[2, 4, 8, 16],
        latent_channels=4,
        sample_size=args.latent_dim,
    )
    lm = Engine(dnn=autoencoder, lr=args.lr, lnp=lnp, wandb_logger=wandb_logger,
                kl_w=args.kl_w, img_size=args.resolution)
    for n,p in lm.named_parameters():
        lnp.lnp(n + ': ' + str(p.data.shape))

    # Callbacks
    # https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html
    lnp.lnp('MAIN callbacks')
    l_callbacks = []

    # model checkpoint
    # https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#automatic-saving
    checkpoint_dirpath = os.path.join(args.output, 'VQVae')
    progress_bar = CustomProgressBar()
    l_callbacks.append(progress_bar)

    if args.resume or args.ckpt_path:
        device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
        if args.ckpt_path is None:
            best_checkpoint = get_elite(os.listdir(checkpoint_dirpath))
            if os.path.exists(os.path.join(checkpoint_dirpath, best_checkpoint)):
                ckp = torch.load(os.path.join(checkpoint_dirpath, best_checkpoint), map_location=device)
                lm.dnn.load_state_dict(ckp.get('dnn'))
            else:
                print(f'No checkpoint found at {os.path.join(checkpoint_dirpath, best_checkpoint)}')
        else:
            ckp = torch.load(args.ckpt_path, map_location=device)
            lm.dnn.load_state_dict(ckp.get('dnn'))

    cbModelCheckpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        dirpath=checkpoint_dirpath,
        filename="epoch_{epoch:05d}-loss_{val_loss:.5f}",
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
    dm = GenEMNISTDataModule(batch_size=args.batch_size, val_split=args.val_split, 
                             num_workers=args.num_workers, num_classes=args.num_classes, 
                             generator=generator, train_size=args.train_size, 
                             test_size=args.test_size, z_dim=args.z_dim,)

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
    parser.add_argument('--experiment', default='VQVae', type=str)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--num-classes', default=47, type=int)
    parser.add_argument('--num-epochs', default=10, type=int)
    parser.add_argument('--val-split', default=0.05, type=float)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--hidden-channels', default=16, type=int)
    parser.add_argument('--num-layers', default=8, type=int)
    parser.add_argument('--latent-dim', default=8, type=int)
    parser.add_argument('--kl-w', default=0.5, type=float)
    parser.add_argument('--generator', type=str)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--z-dim', default=64, type=int)
    parser.add_argument('--lr', default=0.0002, type=float)
    parser.add_argument('--train-size', default=235000, type=int)
    parser.add_argument('--test-size', default=15000, type=int)
    parser.add_argument('--resolution', default=32, type=int)
    parser.add_argument('--ckpt-path', type=str)
    parser.add_argument('--alias', type=str, default=None)

    # trainer level args
    args = parser.parse_args()
    main(args)
