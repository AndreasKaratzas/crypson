import argparse
import os
import logging
from rich.logging import RichHandler
from rich.progress import track
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("rich")

# Set up TensorBoard and wandb
writer = SummaryWriter()
wandb.init(project="conditional-gan-emnist")

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.embedding = nn.Embedding(num_classes, latent_dim)
        self.fc = nn.Linear(latent_dim, 128 * (img_size // 4) ** 2)
        self.res_blocks = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 64),
            AttentionLayer(64),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        embedding = self.embedding(labels)
        z = torch.mul(z, embedding)
        z = self.fc(z)
        z = z.view(z.size(0), 128, self.img_size // 4, self.img_size // 4)
        img = self.res_blocks(z)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.embedding = nn.Embedding(num_classes, img_size * img_size)
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            AttentionLayer(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 1, 4, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        embedding = self.embedding(labels).view(labels.size(0), 1, self.img_size, self.img_size)
        img = torch.cat([img, embedding], dim=1)
        return self.model(img)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = self.relu(x)
        return x

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, in_channels):
        super(AttentionLayer, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out

# Training loop
def train(generator, discriminator, dataloader, criterion, optimizer_G, optimizer_D, device, num_epochs, log_interval, sample_interval):
    for epoch in track(range(num_epochs), description=f"Epoch {epoch+1}/{num_epochs}"):
        for i, (imgs, labels) in enumerate(dataloader):
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            z = torch.randn(imgs.size(0), generator.latent_dim).to(device)
            fake_imgs = generator(z, labels)
            real_preds = discriminator(real_imgs, labels)
            fake_preds = discriminator(fake_imgs.detach(), labels)
            d_loss = criterion(real_preds, torch.ones_like(real_preds)) + criterion(fake_preds, torch.zeros_like(fake_preds))
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_preds = discriminator(fake_imgs, labels)
            g_loss = criterion(fake_preds, torch.ones_like(fake_preds))
            g_loss.backward()
            optimizer_G.step()

            if i % log_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(dataloader)}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")
                writer.add_scalar("D_loss", d_loss.item(), epoch * len(dataloader) + i)
                writer.add_scalar("G_loss", g_loss.item(), epoch * len(dataloader) + i)
                wandb.log({"D_loss": d_loss.item(), "G_loss": g_loss.item()})

            if (epoch + 1) % sample_interval == 0:
                generator.eval()
                with torch.no_grad():
                    fake_imgs = generator(z, labels)
                    grid = make_grid(fake_imgs, nrow=8, normalize=True)
                    save_image(grid, f"generated_images_{epoch+1}.png")
                    writer.add_image("Generated Images", grid, epoch + 1)
                    wandb.log({"Generated Images": wandb.Image(grid)})
                generator.train()

# Evaluation
def evaluate(generator, dataloader, device):
    generator.eval()
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            labels = labels.to(device)
            z = torch.randn(imgs.size(0), generator.latent_dim).to(device)
            fake_imgs = generator(z, labels)
            grid = make_grid(fake_imgs, nrow=8, normalize=True)
            save_image(grid, f"evaluation_{i+1}.png")
            writer.add_image(f"Evaluation {i+1}", grid)
            wandb.log({f"Evaluation {i+1}": wandb.Image(grid)})
    generator.train()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset = datasets.EMNIST(root="data", split="balanced", train=True, download=True, transform=transform)
    test_dataset = datasets.EMNIST(root="data", split="balanced", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    generator = Generator(args.latent_dim, len(train_dataset.classes), args.img_size).to(device)
    discriminator = Discriminator(len(train_dataset.classes), args.img_size).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    if not os.path.exists("generated_images"):
        os.makedirs("generated_images")

    train(generator, discriminator, train_loader, criterion, optimizer_G, optimizer_D, device, args.epochs, args.log_interval, args.sample_interval)
    evaluate(generator, test_loader, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=32, help="Size of the generated images")
    parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of the latent space")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--log-interval", type=int, default=100, help="Interval for logging")
    parser.add_argument("--sample-interval", type=int, default=10, help="Interval for generating sample images")
    args = parser.parse_args()

    main(args)
