
import os
import torch
import wandb
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from torchmetrics import ConfusionMatrix, Accuracy
from lightning.pytorch import LightningModule


class Engine(LightningModule):

    def __init__(self, classifier, lr=1e-3,
                 lnp=None, wandb_logger=None,
                 num_classes=47, ):
        super().__init__()
        self.save_hyperparameters(ignore=['classifier' 'lnp', 'wandb_logger', ])
        self.classifier = classifier
        self.lr = lr
        self.lnp = lnp
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(num_classes=num_classes)
        self.val_acc = Accuracy(num_classes=num_classes)
        self.test_acc = Accuracy(num_classes=num_classes)
        self.test_cm = ConfusionMatrix(num_classes=num_classes)
        self.wandb_logger = wandb_logger

    def forward(self, latents):
        return self.classifier(latents)
    
    def training_step(self, batch, batch_idx):
        """Training step.
        Parameters
        ----------
        batch : tuple
            Batch of real images and corresponding labels.
        batch_idx : int
            Batch index.
        """
        latents, y_true = batch
        
        y_hat = self(latents)
        loss = self.criterion(y_hat, y_true)
        self.train_acc(y_hat, y_true)
        
        self.log_dict(
            {'train_loss': loss,
             'train_accuracy': self.train_acc.compute()},
            on_step=False, on_epoch=True, prog_bar=True)

        # Store the step losses in custom lists
        if not hasattr(self, 'train_l_reg'):
            self.train_l_reg = []
        if not hasattr(self, 'train_acc_reg'):
            self.train_acc_reg = []
        self.train_l_reg.append(loss.item())
        self.train_acc_reg.append(self.train_acc.compute())

        # Log the epoch means
        if (batch_idx + 1) % self.trainer.num_training_batches == 0:
            loss_mean = sum(self.train_l_reg) / len(self.train_l_reg)
            acc_mean = sum(self.train_acc_reg) / len(self.train_acc_reg)
            self.lnp.lnp(
                f"Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] "
                f"Loss (mean): {loss_mean:.4f}"
                f"Accuracy (mean): {acc_mean:.4f}"
            )
            self.logger.experiment.add_scalar('loss', loss_mean, global_step=self.global_step)
            self.logger.experiment.add_scalar('accuracy', acc_mean, global_step=self.global_step)
            self.train_l_reg.clear()
            self.train_acc_reg.clear()

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
        latents, y_true = batch
        
        y_hat = self(latents)
        loss = self.criterion(y_hat, y_true)
        self.val_acc(y_hat, y_true)
        
        # Store the step losses in custom lists
        if not hasattr(self, 'val_l_reg'):
            self.val_l_reg = []
        if not hasattr(self, 'val_acc_reg'):
            self.val_acc_reg = []
        self.val_l_reg.append(loss.item())
        self.val_acc_reg.append(self.val_acc.compute())

        # Log the validation loss
        self.log_dict(
            {'val_loss': loss,
             'val_accuracy': self.val_acc.compute()},
            on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step.
        Parameters
        ----------
        batch : tuple
            Batch of real images and corresponding labels.
        batch_idx : int
            Batch index.
        """
        latents, y_true = batch

        y_hat = self(latents)
        loss = self.criterion(y_hat, y_true)
        self.test_acc(y_hat, y_true)
        self.test_cm(y_hat, y_true)

        # Log the test loss
        self.log_dict(
            {'test_loss': loss,
             'test_accuracy': self.test_acc.compute(),
             'test_confusion_matrix': self.test_cm.compute()},
            on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizers.
        """
        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.lr)
        return optimizer

    def on_validation_epoch_end(self):
        val_l_mean = sum(self.val_l_reg) / len(self.val_l_reg)
        val_acc_mean = sum(self.val_acc_reg) / len(self.val_acc_reg)
        self.lnp.lnp(
            f"Validation Epoch [{self.current_epoch+1}/{self.trainer.max_epochs}] "
            f"Val_loss (mean): {val_l_mean:.4f}"
            f"Val_accuracy (mean): {val_acc_mean:.4f}"
        )
        self.val_l_reg.clear()
        self.val_acc_reg.clear()

    def on_save_checkpoint(self, checkpoint):
        """Save Classifier.
        
        Parameters
        ----------
        checkpoint : dict
            Checkpoint dictionary.
            
        Returns
        -------
        dict
            Updated checkpoint dictionary.
        """
        checkpoint['classifier'] = self.classifier.state_dict()
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        """Load Classifier state dictionaries.
        
        Parameters
        ----------
        checkpoint : dict
            Checkpoint dictionary.
            
        Returns
        -------
        dict
            Updated checkpoint dictionary.
        """
        self.classifier.load_state_dict(checkpoint['classifier'])
        return checkpoint
