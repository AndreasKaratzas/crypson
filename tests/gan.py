
import sys
sys.path.append('../')

import argparse

from lightning.pytorch import Trainer

from src.engine import Engine
from src.dataset import EMNISTDataModule
from src.modules import (
    Generator, Discriminator,
    ConvGenerator, ConvDiscriminator,
    UNetGenerator, UNetDiscriminator)


if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Train a GAN model')
    parser.add_argument('--en-cv', action='store_true', help='Enable convolutional layers')
    parser.add_argument('--en-unet', action='store_true', help='Enable U-Net architecture')
    args = parser.parse_args()

    # Initialize the data module
    emnist_data = EMNISTDataModule(data_dir='../data', batch_size=16, 
                                   val_split=0.15, num_workers=8,
                                   image_size=28)

    # Initialize the generator and discriminator
    if args.en_cv:
        generator = ConvGenerator(z_dim=100, img_shape=(1, 28, 28), n_classes=62)
        discriminator = ConvDiscriminator(img_shape=(1, 28, 28), n_classes=62)
    elif args.en_unet:
        generator = UNetGenerator(z_dim=8, img_shape=(1, 28, 28), n_classes=62)
        discriminator = UNetDiscriminator(z_dim=8, img_shape=(1, 28, 28), n_classes=62)
    else:
        generator = Generator(z_dim=100, img_shape=(28, 28), n_classes=62)
        discriminator = Discriminator(img_shape=(28, 28), n_classes=62) 

    # Initialize the GAN module
    gan_model = Engine(generator=generator, discriminator=discriminator, 
                       num_classes=62, z_dim=100 if not args.en_unet else 8, 
                       lr=0.0002, betas=(0.5, 0.999), clip_grad_norm=5.0, 
                       en_cv=args.en_cv or args.en_unet)

    # Initialize a trainer
    trainer = Trainer(max_epochs=10, accelerator='gpu', enable_model_summary=True, 
                      enable_progress_bar=True, num_sanity_val_steps=0)

    # Train the model
    trainer.fit(gan_model, emnist_data)
