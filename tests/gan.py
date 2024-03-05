
import sys
sys.path.append('../')

import os

from pytorch_lightning import Trainer

from src.engine import Engine
from src.dataset import EMNISTDataModule
from src.modules import Generator, Discriminator

if __name__ == "__main__":
    # Initialize the data module
    emnist_data = EMNISTDataModule(data_dir='../data', batch_size=32, 
                                   img_size=32, val_split=0.15, 
                                   num_workers=8)

    # Initialize the generator and discriminator
    generator = Generator(z_dim=100, img_shape=(32, 32), n_classes=62)
    discriminator = Discriminator(img_shape=(32, 32), n_classes=62)

    # Initialize the GAN module
    gan_model = Engine(generator=generator, discriminator=discriminator, 
                       num_classes=62)

    # Initialize a trainer
    trainer = Trainer(max_epochs=50, accelerator='gpu', enable_model_summary=True, 
                    enable_progress_bar=True, num_sanity_val_steps=1)

    # Train the model
    trainer.fit(gan_model, emnist_data)
