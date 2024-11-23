import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gan_model import Discriminator, Generator
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_gan_model(
    dataloader: DataLoader, input_dim: int, latent_dim: int, lr: float, epochs: int, device: torch.device
) -> Generator:
    generator = Generator(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    # Optimizers
    generator_optim = optim.Adam(generator.parameters(), lr=lr)
    discriminator_optim = optim.Adam(discriminator.parameters(), lr=lr)

    # Loss function
    adversarial_loss = nn.BCELoss()

    # Training loop
    for epoch in range(epochs):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in pbar:
            real_samples = data.to(device)
            batch_size = real_samples.size(0)

            # Train Discriminator
            discriminator_optim.zero_grad()
            z = torch.randn(batch_size, latent_dim)
            z.to(device)
            fake_samples = generator(z)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            real_loss = adversarial_loss(discriminator(real_samples), real_labels)
            fake_loss = adversarial_loss(discriminator(fake_samples.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            discriminator_optim.step()

            # Train Generator
            generator_optim.zero_grad()
            g_loss = adversarial_loss(discriminator(fake_samples), real_labels)
            g_loss.backward()
            generator_optim.step()

            if epoch % 100 == 0:
                pbar.set_description(f"Epoch [{epoch}/{epochs}]  D Loss: {d_loss.item()}  G Loss: {g_loss.item()}")
    return generator


def generate_new_samples(generator: Generator, latent_dim: int, num_samples: int, device: torch.device) -> np.array:
    z = torch.randn(num_samples, latent_dim).to(device)
    return generator(z).to("cpu").numpy()


# Hyperparameters
# latent_dim = 100
# batch_size = 64
# epochs = 10000
# lr = 0.0002
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
