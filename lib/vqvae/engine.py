
import os
import torch
import wandb
import torchvision
import torch.nn.functional as F

from collections import deque
from torchvision.utils import save_image
from lightning.pytorch import LightningModule


class Engine(LightningModule):

    def __init__(self, dnn, lr=0.0002, img_size=32,
                 lnp=None, wandb_logger=None, kl_w=0.5):
        super().__init__()
        self.save_hyperparameters(ignore=['dnn' 'lnp', 
                                          'wandb_logger', 'generator'])
        self.dnn = dnn
        self.lr = lr
        self.img_size = img_size
        self.kl_w = kl_w
        self.lnp = lnp
        self.wandb_logger = wandb_logger

    def forward(self, img):
        return self.dnn(img)
    
    def training_step(self, batch, batch_idx):
        """Training step.
        Parameters
        ----------
        batch : tuple
            Batch of real images and corresponding labels.
        batch_idx : int
            Batch index.
        """
        img = batch

        latent_dist = self.dnn.encode(img)
        latents = latent_dist.sample()
        out = self.dnn.decode(latents)
        recon_loss = torch.mean((img - out) ** 2)
        kl_loss = latent_dist.kl().mean()
        loss = recon_loss + kl_loss

        # out, mu, log_var = self(img)
        # recon_loss = F.binary_cross_entropy(out, img, reduction='mean')
        # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # loss = ((1 - self.kl_w) * recon_loss) + ((self.kl_w) * kl_loss)
        
        self.log_dict(
            {'recon_loss': recon_loss,
             'kl_loss': kl_loss,
             'loss': loss},
            on_step=False, on_epoch=True, prog_bar=True)

        # Store the step losses in custom lists
        if not hasattr(self, 'losses'):
            self.losses = []
        self.losses.append(loss.item())

        # Log the epoch means
        if (batch_idx + 1) % self.trainer.num_training_batches == 0:
            loss_mean = sum(self.losses) / len(self.losses)
            self.lnp.lnp(
                f"Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] "
                f"Loss (mean): {loss_mean:.4f}"
            )
            self.logger.experiment.add_scalar('loss', loss_mean, global_step=self.global_step)
            self.losses.clear()

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.
        Parameters
        ----------
        batch : tuple
            Batch of real images and corresponding labels.
        batch_idx : int
            Batch index.
        """
        img = batch
        
        # out, mu, log_var = self(img)
        # recon_loss = F.binary_cross_entropy(out, img, reduction='mean')
        # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # val_loss = ((1 - self.kl_w) * recon_loss) + ((self.kl_w) * kl_loss)
        latent_dist = self.dnn.encode(img)
        latents = latent_dist.sample()
        out = self.dnn.decode(latents)
        recon_loss = torch.mean((img - out) ** 2)
        kl_loss = latent_dist.kl().mean()
        val_loss = recon_loss + kl_loss

        # Store the step losses in custom lists
        if not hasattr(self, 'val_loss'):
            self.val_loss = []
        self.val_loss.append(val_loss.item())

        if not hasattr(self, 'reconstructed_images'):
            self.reconstructed_images = deque(maxlen=64)

        if len(self.reconstructed_images) < 64:
            for i in range(64 - len(self.reconstructed_images)):
                self.reconstructed_images.append(out[i].detach().cpu())

        # Log the validation loss
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step.
        Parameters
        ----------
        batch : tuple
            Batch of real images and corresponding labels.
        batch_idx : int
            Batch index.
        """
        img = batch
        
        # out, mu, log_var = self(img)
        # recon_loss = F.binary_cross_entropy(out, img, reduction='mean')
        # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # test_loss = ((1 - self.kl_w) * recon_loss) + ((self.kl_w) * kl_loss)
        latent_dist = self.dnn.encode(img)
        latents = latent_dist.sample()
        out = self.dnn.decode(latents)
        recon_loss = torch.mean((img - out) ** 2)
        kl_loss = latent_dist.kl().mean()
        test_loss = recon_loss + kl_loss

        # Store the step losses in custom lists
        if not hasattr(self, 'test_loss'):
            self.test_loss = []
        self.test_loss.append(test_loss.item())

        # Log the test loss
        self.log('test_loss', test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizers.
        """
        optimizer = torch.optim.AdamW(self.dnn.parameters(), lr=3e-4)
        return optimizer

    def on_validation_epoch_end(self):
        val_loss_mean = sum(self.val_loss) / len(self.val_loss)
        self.lnp.lnp(
            f"Validation Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] "
            f"Val_loss (mean): {val_loss_mean:.4f}"
        )
        self.val_loss.clear()

        image_dir = os.path.join(self.trainer.log_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(
            image_dir, f"reconstructed_images_epoch_{self.current_epoch}.png")
        
        reconstructed_images = torch.stack(list(self.reconstructed_images))
        reconstructed_images = reconstructed_images.view(
            -1, 1, self.img_size, self.img_size).permute(0, 1, 3, 2)
        grid = torchvision.utils.make_grid(
            reconstructed_images, nrow=5, normalize=True)
        self.reconstructed_images.clear()
        save_image(grid, image_path)
        self.logger.experiment.add_image(
            "reconstructed_images", grid, global_step=self.global_step)
        self.wandb_logger.experiment.log(
            {"reconstructed_images": wandb.Image(image_path)}, step=self.global_step)
        self.lnp.lnp(
            f"Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] Reconstructed Images")
    
    def on_save_checkpoint(self, checkpoint):
        """Save AutoEncoder.
        
        Parameters
        ----------
        checkpoint : dict
            Checkpoint dictionary.
            
        Returns
        -------
        dict
            Updated checkpoint dictionary.
        """
        checkpoint['dnn'] = self.dnn.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """Load AutoEncoder state dictionaries.
        
        Parameters
        ----------
        checkpoint : dict
            Checkpoint dictionary.
            
        Returns
        -------
        dict
            Updated checkpoint dictionary.
        """
        self.dnn.load_state_dict(checkpoint['dnn'])
        return checkpoint
