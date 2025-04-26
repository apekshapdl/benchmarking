import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta=1.0):
        super(BetaVAE, self).__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

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

    def loss_function(self, x, x_hat, mu, logvar):
        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

def train_beta_vae(model, dataloader, optimizer, device='cpu', epochs=50):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (x_batch,) in dataloader:
            x_batch = x_batch.to(device)
            x_hat, mu, logvar, _ = model(x_batch)
            loss, _, _ = model.loss_function(x_batch, x_hat, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        #print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

def evaluate_beta_vae(model, x_tensor):
    model.eval()
    recon_errors = []

    with torch.no_grad():
        for x in x_tensor:
            x = x.unsqueeze(0)
            x_hat, _, _, _ = model(x)
            error = F.mse_loss(x_hat, x, reduction='none').mean().item()
            recon_errors.append(error)

    return np.array(recon_errors)
