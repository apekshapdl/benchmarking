
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np

# ---------- FactorVAE Main Model ----------
class FactorVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(FactorVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
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


# ---------- Discriminator for TC Estimation ----------
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2)  # Binary: real (0) or permuted (1)
        )

    def forward(self, z):
        return self.net(z)

# Reconstruction + KL + Total Correlation (via discriminator)
def factor_vae_loss(x, x_hat, mu, logvar, z, D_z, beta):
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # Total Correlation (using log-odds from discriminator)
    logits = D_z(z)
    tc_estimate = (logits[:, 0] - logits[:, 1]).mean()  # real - permuted
    total_corr = beta * tc_estimate

    return recon_loss + kl + total_corr, recon_loss, kl, total_corr


def permute_dims(z):
    B, D = z.size()
    z_perm = []
    for d in range(D):
        z_perm.append(z[:, d][torch.randperm(B)])
    return torch.stack(z_perm, dim=1)



# Train Function
def train_factor_vae(model, discriminator, train_loader, optimizer_vae, optimizer_disc, beta=10.0, epochs=50):
    model.train()
    discriminator.train()

    for epoch in range(epochs):
        for (x_batch,) in train_loader:
            #VAE
            x_hat, mu, logvar, z = model(x_batch)
            z_perm = permute_dims(z).detach()
            D_z = lambda zz: discriminator(zz)

            loss, recon, kl, tc = factor_vae_loss(x_batch, x_hat, mu, logvar, z, D_z, beta)
            optimizer_vae.zero_grad()
            loss.backward()
            optimizer_vae.step()

            #Discriminator
            D_real = discriminator(z.detach())
            D_perm = discriminator(z_perm)

            logits = torch.cat([D_real, D_perm], dim=0)
            labels = torch.cat([
                torch.zeros(D_real.size(0), dtype=torch.long),
                torch.ones(D_perm.size(0), dtype=torch.long)
            ], dim=0).to(x_batch.device)

            loss_disc = F.cross_entropy(logits, labels)
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step()



# Evaluate Function
def evaluate_factor_vae(model, x_tensor):
    model.eval()
    recon_errors = []

    with torch.no_grad():
        for x in x_tensor:
            x = x.unsqueeze(0)
            x_hat, _, _, _ = model(x)
            error = F.mse_loss(x_hat, x, reduction='none').mean().item()
            recon_errors.append(error)

    return np.array(recon_errors)

#print(" fvae.py loaded successfully")
#print("Available symbols:", dir())
