import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, num_classes):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        emb = self.label_emb(labels)
        noise = noise.view(noise.size(0), -1)
        gen_input = torch.mul(emb, noise)
        gen_input = gen_input.view(gen_input.size(0), -1, 1, 1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img
    

class Discriminator(nn.Module):
    def __init__(self, img_size, num_classes):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, img_size*img_size)
        self.model = nn.Sequential(
            nn.Linear(img_size*img_size*2, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        img_flat = img.view(img.size(0), -1)
        label_emb = self.label_emb(labels).squeeze()
        d_in = torch.cat((img_flat, label_emb), -1)
        validity = self.model(d_in)
        return validity


# Set the parameters
latent_dim = 100
img_size = 32
num_classes = 62
batch_size = 64
num_epochs = 50
lr = 0.0002
beta1 = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the EMNIST dataset
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
dataset = EMNIST(root='./data', split='balanced', train=True,
                 transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create instances of the generator and discriminator
generator = Generator(latent_dim, img_size, num_classes).to(device)
discriminator = Discriminator(img_size, num_classes).to(device)

# Define the loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(),
                         lr=lr, betas=(beta1, 0.999))

# Training loop
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # Train the discriminator
        optimizer_D.zero_grad()
        real_validity = discriminator(real_imgs, labels)
        real_labels = torch.ones_like(real_validity, device=device)
        real_loss = criterion(real_validity, real_labels)

        noise = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)
        fake_imgs = generator(noise, labels)
        fake_validity = discriminator(fake_imgs.detach(), labels)
        fake_labels = torch.zeros_like(fake_validity, device=device)
        fake_loss = criterion(fake_validity, fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train the generator
        optimizer_G.zero_grad()
        fake_validity = discriminator(fake_imgs, labels)
        g_loss = criterion(fake_validity, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Print progress
        if (i + 1) % 100 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Save the trained models
torch.save(generator.state_dict(), "generator.pt")
torch.save(discriminator.state_dict(), "discriminator.pt")

# Evaluation loop
generator.eval()
with torch.no_grad():
    for i in range(num_classes):
        label = torch.tensor([i], device=device)
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        generated_img = generator(noise, label)
        generated_img = generated_img.squeeze().cpu().numpy()
        plt.imshow(generated_img, cmap='gray')
        plt.title(f"Generated Image - Label {i}")
        plt.axis('off')
        plt.show()

# Testing loop with custom labels
generator.eval()
with torch.no_grad():
    custom_labels = [10, 25, 50]  # Example custom labels (A, Z, 2)
    for label in custom_labels:
        label_tensor = torch.tensor([label], device=device)
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        generated_img = generator(noise, label_tensor)
        generated_img = generated_img.squeeze().cpu().numpy()
        plt.imshow(generated_img, cmap='gray')
        plt.title(f"Generated Image - Custom Label {label}")
        plt.axis('off')
        plt.show()
