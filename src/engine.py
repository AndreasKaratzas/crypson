
import torch
import torchvision

from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.autograd import Variable


class Engine(LightningModule):
    """Conditional Generative Adversarial Network (GAN) model.
    
    Parameters
    ----------
    generator : torch.nn.Module
        Generator model.
    discriminator : torch.nn.Module
        Discriminator model.
    z_dim : int, optional
        Dimension of the noise vector.
    lr : float, optional
        Learning rate for the optimizer.
    betas : tuple, optional
        Coefficients for computing running averages of gradient and its square.
    """
    
    def __init__(self, generator, discriminator, num_classes, z_dim=100, lr=0.0002, betas=(0.5, 0.999)):
        super().__init__()
        self.save_hyperparameters(ignore=['generator', 'discriminator'])
        self.generator = generator
        self.discriminator = discriminator
        self.z_dim = z_dim
        self.lr = lr
        self.betas = betas
        self.num_classes = num_classes
        self.automatic_optimization = False
        self.validation_z = torch.randn(8, self.z_dim, device=self.device)
        self.criterion = torch.nn.BCELoss()

    def forward(self, z, labels):
        return self.generator(z, labels)
    
    def training_step(self, batch, batch_idx):
        real_images, labels = batch
        real_images = real_images.squeeze(1)
        optimizer_g, optimizer_d = self.optimizers()

        # Sample noise
        z = torch.randn(real_images.size(0), self.z_dim, device=self.device)
        # Generate fake labels
        fake_labels = Variable(torch.randint(0, self.num_classes, (real_images.size(0),))).to(self.device)
        # Generate fake images
        self.toggle_optimizer(optimizer_g)
        fake_images = self(z, fake_labels)
        
        # Generator training
        fake_pred = self.discriminator(fake_images, fake_labels)
        g_loss = self.criterion(fake_pred, Variable(torch.ones_like(fake_pred)).to(self.device))
        self.log('g_loss', g_loss)
        # Backward pass
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        # Discriminator training
        real_pred = self.discriminator(real_images, labels)
        real_loss = self.criterion(real_pred, Variable(torch.ones_like(real_pred)).to(self.device))
        fake_labels = Variable(torch.randint(
            0, self.num_classes, (real_images.size(0),))).to(self.device)
        fake_images = self(z, fake_labels)
        fake_pred = self.discriminator(fake_images, fake_labels)
        fake_loss = self.criterion(fake_pred, Variable(torch.zeros_like(fake_pred)).to(self.device))
        d_loss = (real_loss + fake_loss) / 2
        self.log('d_loss', d_loss)
        # Backward pass
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def validation_step(self, batch, batch_idx):
        real_images, labels = batch
        real_images = real_images.squeeze(1)
        
        # Generating fake images
        z = torch.randn(real_images.size(0), self.z_dim, device=self.device)
        fake_images = self(z, labels)

        # Discriminator predictions
        real_pred = self.discriminator(real_images, labels)
        fake_pred = self.discriminator(fake_images, labels)
        
        # Calculate loss for real and fake images
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        d_loss = (real_loss + fake_loss) / 2
        
        # Log the validation loss
        self.log('val_d_loss', d_loss)
    
    def configure_optimizers(self):
        opt_d = Adam(self.discriminator.parameters(),
                     lr=self.lr, betas=self.betas)
        opt_g = Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        return [opt_d, opt_g], []
    
    def on_validation_epoch_end(self):
        # Log sampled images
        sample_images = self(self.validation_z.to(
            self.device), torch.randint(0, 10, (8,)).to(self.device))
        # Unsqueeze the images to 3D
        sample_images = sample_images.unsqueeze(1)
        grid = torchvision.utils.make_grid(sample_images)
        self.logger.experiment.add_image(
            "generated_images", grid, self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['generator'] = self.generator.state_dict()
        checkpoint['discriminator'] = self.discriminator.state_dict()
        return checkpoint
    
    def on_load_checkpoint(self, checkpoint):
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        return checkpoint
