
import os
import torch
import wandb
import numpy as np
import torchvision

from lightning.pytorch import LightningModule
from torch.optim import Adam
from torchvision.utils import save_image


class Engine(LightningModule):
    """Conditional Generative Adversarial Network (GAN) model.
    
    Parameters
    ----------
    generator : torch.nn.Module
        Generator model.
    discriminator : torch.nn.Module
        Discriminator model.
    num_classes : int
        Number of classes for conditional discrimination.
    z_dim : int, optional
        Dimension of the noise vector. Default is 100.
    lr : float, optional
        Learning rate for the optimizer. Default is 0.0002.
    betas : tuple, optional
        Coefficients for computing running averages of gradient and its square. Default is (0.5, 0.999).
    clip_grad_norm : float, optional
        Value for gradient clipping. Default is 5.0.
    lnp : object, optional
        Log and print utility. Default is None.
    wandb_logger : object, optional
        Weights and Biases logger. Default is None.
    """

    def __init__(self, generator, discriminator, num_classes,
                 z_dim=100, lr=0.0002, betas=(0.5, 0.999),
                 lnp=None, wandb_logger=None):
        super().__init__()
        self.save_hyperparameters(ignore=['generator', 'discriminator', 'lnp', 'wandb_logger'])
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.lr = lr
        self.betas = betas
        self.num_classes = num_classes
        self.automatic_optimization = False
        self.validation_z = torch.randn(self.num_classes, self.z_dim, device=self.device)
        self.criterion = torch.nn.BCELoss()
        self.lnp = lnp
        self.wandb_logger = wandb_logger

    def forward(self, z, labels):
        """Forward pass through the generator.
        
        Parameters
        ----------
        z : torch.Tensor
            Noise vector.
        labels : torch.Tensor
            Class labels.
            
        Returns
        -------
        torch.Tensor
            Generated images.
        """
        return self.generator(z, labels)

    def training_step(self, batch, batch_idx):
        """Training step.
        Parameters
        ----------
        batch : tuple
            Batch of real images and corresponding labels.
        batch_idx : int
            Batch index.
        """
        real_images, labels = batch
        opt_d, opt_g = self.optimizers()

        # Discriminator training
        self.toggle_optimizer(opt_d)
        real_validity = self.discriminator(real_images, labels)
        real_labels = torch.ones_like(real_validity, device=self.device)
        real_loss = self.criterion(real_validity, real_labels)
        noise = torch.randn(real_images.size(0), self.z_dim, 1, 1, device=self.device)
        fake_imgs = self(noise, labels)
        fake_validity = self.discriminator(fake_imgs.detach(), labels)
        fake_labels = torch.zeros_like(fake_validity, device=self.device)
        fake_loss = self.criterion(fake_validity, fake_labels)
        d_loss = real_loss + fake_loss
        # Update the discriminator
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)
        self.log('d_loss', d_loss, on_step=False, on_epoch=True, prog_bar=True)

        
        # Generator training
        self.toggle_optimizer(opt_g)
        noise = torch.randn(real_images.size(0), self.z_dim, 1, 1, device=self.device)
        fake_imgs = self(noise, labels)
        fake_validity = self.discriminator(fake_imgs, labels)
        g_loss = self.criterion(fake_validity, real_labels)
        # Update the generator
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)
        self.log('g_loss', g_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store the step losses in custom lists
        if not hasattr(self, 'd_losses'):
            self.d_losses = []
        if not hasattr(self, 'g_losses'):
            self.g_losses = []
        self.d_losses.append(d_loss.item())
        self.g_losses.append(g_loss.item())

        # Log the epoch means
        if (batch_idx + 1) % self.trainer.num_training_batches == 0:
            d_loss_mean = sum(self.d_losses) / len(self.d_losses)
            g_loss_mean = sum(self.g_losses) / len(self.g_losses)
            global_step = self.global_step  # Get the current global step
            self.lnp.lnp(
                f"Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] "
                f"D_loss (mean): {d_loss_mean:.4f}, G_loss (mean): {g_loss_mean:.4f}"
            )
            self.logger.experiment.add_scalar('d_loss', d_loss_mean, global_step=global_step)
            self.logger.experiment.add_scalar('g_loss', g_loss_mean, global_step=global_step)
            # Clear the losses for the next epoch
            self.d_losses.clear()
            self.g_losses.clear()

    def validation_step(self, batch, batch_idx):
        """Validation step.
        Parameters
        ----------
        batch : tuple
            Batch of real images and corresponding labels.
        batch_idx : int
            Batch index.
        """
        real_images, labels = batch
        # Generating fake images
        noise = torch.randn(real_images.size(0), self.z_dim, 1, 1, device=self.device)
        fake_imgs = self(noise, labels)
        # Discriminator loss
        real_validity = self.discriminator(real_images, labels)
        real_labels = torch.ones_like(real_validity, device=self.device)
        real_loss = self.criterion(real_validity, real_labels)
        fake_validity = self.discriminator(fake_imgs.detach(), labels)
        fake_labels = torch.zeros_like(fake_validity, device=self.device)
        fake_loss = self.criterion(fake_validity, fake_labels)
        val_loss = real_loss + fake_loss

        # Store the step losses in custom lists
        if not hasattr(self, 'val_loss'):
            self.val_loss = []
        self.val_loss.append(val_loss.item())

        # Log the validation loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator.
        
        Returns
        -------
        list
            List of optimizers for generator and discriminator.
        """
        opt_d = Adam(self.discriminator.parameters(),
                     lr=self.lr, betas=self.betas)
        opt_g = Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        return [opt_d, opt_g]

    def on_validation_epoch_end(self):
        """Log sampled images at the end of each validation epoch."""
        # Log sampled images
        val_loss_mean = sum(self.val_loss) / len(self.val_loss)
        self.lnp.lnp(
            f"Validation Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] "
            f"Val_loss (mean): {val_loss_mean:.4f}"
        )
        self.val_loss.clear()
        
        image_dir = os.path.join(self.trainer.log_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"generated_images_epoch_{self.current_epoch}.png")

        # Generate images
        generated_images = self(self.validation_z, torch.arange(self.num_classes, device=self.device))
        grid = torchvision.utils.make_grid(generated_images, nrow=5, normalize=True)        
        save_image(grid, image_path)
        self.logger.experiment.add_image("generated_images", grid, global_step=self.global_step)
        self.wandb_logger.experiment.log({"generated_images": wandb.Image(image_path)}, step=self.global_step)
        self.lnp.lnp(f"Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] Generated Images")

    def on_save_checkpoint(self, checkpoint):
        """Save generator and discriminator state dictionaries.
        
        Parameters
        ----------
        checkpoint : dict
            Checkpoint dictionary.
            
        Returns
        -------
        dict
            Updated checkpoint dictionary.
        """
        checkpoint['z_dim'] = self.z_dim
        checkpoint['num_classes'] = self.num_classes
        checkpoint['lr'] = self.lr
        checkpoint['betas'] = self.betas
        checkpoint['generator'] = self.generator.state_dict()
        checkpoint['discriminator'] = self.discriminator.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """Load generator and discriminator state dictionaries.
        
        Parameters
        ----------
        checkpoint : dict
            Checkpoint dictionary.
            
        Returns
        -------
        dict
            Updated checkpoint dictionary.
        """
        self.z_dim = checkpoint['z_dim']
        self.num_classes = checkpoint['num_classes']
        self.lr = checkpoint['lr']
        self.betas = checkpoint['betas']
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        return checkpoint
