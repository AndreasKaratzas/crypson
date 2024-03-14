
import os
import torch
import wandb
import torchvision

from collections import deque
from torchvision.utils import save_image
from lightning.pytorch import LightningModule


class Engine(LightningModule):

    def __init__(self, dnn, num_classes,
                 z_dim=64, lr=0.0002,
                 lnp=None, wandb_logger=None,
                 codebook_size=2 ** 8, 
                 entropy_loss_weight=0.02, 
                 diversity_gamma=1.):
        super().__init__()
        self.save_hyperparameters(ignore=['dnn' 'lnp', 
                                          'wandb_logger', 'generator'])
        self.dnn = dnn
        self.z_dim = z_dim
        self.lr = lr
        self.codebook_size = codebook_size
        self.num_classes = num_classes
        self.entrop_loss_weight = entropy_loss_weight
        self.diversity_gamma = diversity_gamma
        self.criterion = torch.nn.MSELoss()
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

        out, indices, entropy_aux_loss = self.dnn(img)
        recon_loss = self.criterion(out, img)
        loss = recon_loss + entropy_aux_loss

        self.log_dict(
            {'recon_loss': recon_loss, 'entropy_aux_loss': entropy_aux_loss, 
             'active_percentage': indices.unique().numel() / self.codebook_size * 100},
            on_step=False, on_epoch=True, prog_bar=True)

        # Store the step losses in custom lists
        if not hasattr(self, 'recon_losses'):
            self.recon_losses = []
        if not hasattr(self, 'entropy_aux_losses'):
            self.entropy_aux_losses = []
        if not hasattr(self, 'active_percentages'):
            self.active_percentages = []
        self.recon_losses.append(recon_loss.item())
        self.entropy_aux_losses.append(entropy_aux_loss.item())
        self.active_percentages.append(indices.unique().numel() / self.codebook_size * 100)

        # Log the epoch means
        if (batch_idx + 1) % self.trainer.num_training_batches == 0:
            recon_loss_mean = sum(self.recon_losses) / len(self.recon_losses)
            entropy_aux_loss_mean = sum(self.entropy_aux_losses) / len(self.entropy_aux_losses)
            active_percentage_mean = sum(self.active_percentages) / len(self.active_percentages)
            self.lnp.lnp(
                f"Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] "
                f"Recon_loss (mean): {recon_loss_mean:.4f}, "
                f"Entropy_aux_loss (mean): {entropy_aux_loss_mean:.4f}"
                f"Active %: {active_percentage_mean:.4f}"
            )
            self.logger.experiment.add_scalar('recon_loss', recon_loss_mean, global_step=self.global_step)
            self.logger.experiment.add_scalar('entropy_aux_loss', entropy_aux_loss_mean, global_step=self.global_step)
            self.logger.experiment.add_scalar('active_percentage', active_percentage_mean, global_step=self.global_step)
            self.recon_losses.clear()
            self.entropy_aux_losses.clear()
            self.active_percentages.clear()

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
        
        out, indices, entropy_aux_loss = self.dnn(img)
        recon_loss = self.criterion(out, img)
        val_loss = recon_loss + entropy_aux_loss

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
        
        out, indices, entropy_aux_loss = self.dnn(img)
        recon_loss = self.criterion(out, img)
        test_loss = recon_loss + entropy_aux_loss

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
        reconstructed_images = reconstructed_images.view(-1, 1, 32, 32).permute(0, 1, 3, 2)
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
