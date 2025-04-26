import torch
import torch.nn as nn
import torch.nn.functional as F
from vae_models.utils import compute_recon_error

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10, cond_dim=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def encode(self, x, y):
        x_cond = torch.cat([x, y], dim=1)
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        z_cond = torch.cat([z, y], dim=1)
        return self.decoder(z_cond)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar, z

    def loss_function(self, x, x_hat, mu, logvar):
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return recon + kl, recon, kl


def train_conditional_vae(model, dataloader, optimizer, device='cpu', epochs=50):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for (x_batch, y_batch) in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x_hat, mu, logvar, _ = model(x_batch, y_batch)
            loss, _, _ = model.loss_function(x_batch, x_hat, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        #print(f"[CVAE] Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")


def evaluate_conditional_vae(model, x_tensor, y_tensor, device='cpu'):
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_tensor = x_tensor.to(device)
        y_tensor = y_tensor.to(device)

        x_hat, _, _, z= model(x_tensor, y_tensor)
        errors = compute_recon_error(x_tensor, x_hat)
    return errors.cpu().numpy(), z.cpu().numpy()
