import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

class BetaTCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(BetaTCVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

def permute_dims(z):
    B, D = z.size()
    z_perm = []
    for d in range(D):
        z_perm.append(z[:, d][torch.randperm(B)])
    return torch.stack(z_perm, dim=1)

def tcvae_loss(x, x_hat, mu, logvar, z, z_perm, D, alpha=1.0, beta=6.0, gamma=1.0):
    # Reconstruction loss
    recon = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    
    # KL divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # Total Correlation estimation via Discriminator
    D_z = D(z)
    D_z_perm = D(z_perm)
    tc_estimate = (D_z[:, 0] - D_z[:, 1]).mean()

    # Combine losses with scaling
    loss = recon + alpha * kl + beta * tc_estimate  

    return loss, recon, kl, tc_estimate

class TCVDiscriminator(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2)
        )

    def forward(self, z):
        return self.net(z)

def train_tcvae(model, discriminator, dataloader, optimizer_vae, optimizer_disc, alpha=1.0, beta=6.0, epochs=50):
    model.train()
    discriminator.train()

    for epoch in range(epochs):
        for (x_batch,) in dataloader:
            x_hat, mu, logvar, z = model(x_batch)
            z_perm = permute_dims(z).detach()

            # VAE + TC Loss
            loss, recon, kl, tc = tcvae_loss(x_batch, x_hat, mu, logvar, z, z_perm, discriminator, alpha, beta)
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            # Discriminator update
            D_real = discriminator(z.detach())
            D_fake = discriminator(z_perm)

            D_logits = torch.cat([D_real, D_fake], dim=0)
            labels = torch.cat([
                torch.zeros(D_real.size(0), dtype=torch.long),
                torch.ones(D_fake.size(0), dtype=torch.long)
            ], dim=0)

            loss_disc = F.cross_entropy(D_logits, labels)
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()

def evaluate_tcvae(model, x_tensor):
    model.eval()
    errors = []

    with torch.no_grad():
        for x in x_tensor:
            x = x.unsqueeze(0) 
            x_hat, _, _, z = model(x)
            error = F.mse_loss(x_hat, x, reduction='none').mean().item()
            errors.append(error)

    return np.array(errors) 

