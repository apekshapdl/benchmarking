import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc as sk_auc


# === Reparameterization ===
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# === Encoders ===
class EncoderZ3(nn.Module):
    def __init__(self, input_dim, z3_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, z3_dim)
        self.fc_logvar = nn.Linear(256, z3_dim)

    def forward(self, x):
        h = F.relu(self.fc(x))
        return self.fc_mu(h), self.fc_logvar(h)

class EncoderZ2(nn.Module):
    def __init__(self, input_dim, z3_dim, z2_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim + z3_dim, 256)
        self.fc_mu = nn.Linear(256, z2_dim)
        self.fc_logvar = nn.Linear(256, z2_dim)

    def forward(self, x, z3):
        h = F.relu(self.fc(torch.cat([x, z3], dim=-1)))
        return self.fc_mu(h), self.fc_logvar(h)

class EncoderZ1(nn.Module):
    def __init__(self, input_dim, z2_dim, z3_dim, z1_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim + z2_dim + z3_dim, 256)
        self.fc_mu = nn.Linear(256, z1_dim)
        self.fc_logvar = nn.Linear(256, z1_dim)

    def forward(self, x, z2, z3):
        h = F.relu(self.fc(torch.cat([x, z2, z3], dim=-1)))
        return self.fc_mu(h), self.fc_logvar(h)

# === Decoder ===
class DeepDecoder(nn.Module):
    def __init__(self, z1_dim, z2_dim, z3_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(z1_dim + z2_dim + z3_dim, 256)
        self.fc_out = nn.Linear(256, output_dim)

    def forward(self, z1, z2, z3):
        h = F.relu(self.fc(torch.cat([z1, z2, z3], dim=-1)))
        return self.fc_out(h)

# === Deep Hierarchical VAE ===
class DeepHierarchicalVAE(nn.Module):
    def __init__(self, input_dim, z1_dim, z2_dim, z3_dim):
        super().__init__()
        self.encoder_z3 = EncoderZ3(input_dim, z3_dim)
        self.encoder_z2 = EncoderZ2(input_dim, z3_dim, z2_dim)
        self.encoder_z1 = EncoderZ1(input_dim, z2_dim, z3_dim, z1_dim)
        self.decoder = DeepDecoder(z1_dim, z2_dim, z3_dim, input_dim)

    def forward(self, x):
        mu_z3, logvar_z3 = self.encoder_z3(x)
        z3 = reparameterize(mu_z3, logvar_z3)

        mu_z2, logvar_z2 = self.encoder_z2(x, z3)
        z2 = reparameterize(mu_z2, logvar_z2)

        mu_z1, logvar_z1 = self.encoder_z1(x, z2, z3)
        z1 = reparameterize(mu_z1, logvar_z1)

        x_hat = self.decoder(z1, z2, z3)
        return x_hat, mu_z1, logvar_z1, mu_z2, logvar_z2, mu_z3, logvar_z3

# === Loss ===
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def loss_function(x, x_hat, mu1, logvar1, mu2, logvar2, mu3, logvar3, kl_weight=1.0):
    recon = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kl1 = kl_divergence(mu1, logvar1)
    kl2 = kl_divergence(mu2, logvar2)
    kl3 = kl_divergence(mu3, logvar3)
    return recon + kl_weight * (kl1 + kl2 + kl3), recon, kl1, kl2, kl3


def train_models_deep_hvae(X_train, X_test, y_test, input_dim, z1_list, z2_list, z3_list, epoch_list, optimizer_type="adam"):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test = y_test.astype(int)

    train_loader = lambda: DataLoader(TensorDataset(X_train_tensor), batch_size=64, shuffle=True)

    results = []

    for z1_dim in z1_list:
        for z2_dim in z2_list:
            for z3_dim in z3_list:
                for epochs in epoch_list:
                    model = DeepHierarchicalVAE(input_dim, z1_dim, z2_dim, z3_dim)

                    if optimizer_type.lower() == "adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    elif optimizer_type.lower() == "sgd":
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    else:
                        raise ValueError("Unsupported optimizer")

                    model.train()
                    for epoch in range(epochs):
                        kl_weight = min(1.0, epoch / 50)
                        for (x_batch,) in train_loader():
                            x_hat, mu1, logvar1, mu2, logvar2, mu3, logvar3 = model(x_batch)
                            loss, _, _, _, _ = loss_function(x_batch, x_hat, mu1, logvar1, mu2, logvar2, mu3, logvar3, kl_weight)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    recon_errors, roc_auc = evaluate_model_deep_hvae(model, X_test_tensor, y_test)
                    precision, recall, _ = precision_recall_curve(y_test, recon_errors)
                    pr_auc = sk_auc(recall, precision)
                    results.append({
                        "z1_dim": z1_dim,
                        "z2_dim": z2_dim,
                        "z3_dim": z3_dim,
                        "epochs": epochs,
                        "optimizer": optimizer_type,
                        "roc_auc": roc_auc,
                        "pr_auc" : pr_auc,
                        "model": model
                    })

                    print(f"z1: {z1_dim}, z2: {z2_dim}, z3: {z3_dim}, epochs: {epochs}, ROC AUC: {roc_auc:.4f}, PR AUC : {pr_auc:.4f}")

    return results


def evaluate_model_deep_hvae(model, X_test_tensor, y_test, batch_size=128):
    model.eval()
    recon_errors = []

    with torch.no_grad():
        loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size)
        for batch in loader:
            x = batch[0]
            x_hat, *_ = model(x)
            batch_errors = F.mse_loss(x_hat, x, reduction='none').mean(dim=1).cpu().numpy()
            recon_errors.extend(batch_errors)

    recon_errors = np.array(recon_errors)
    recon_errors = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)

    if np.isnan(recon_errors).any():
        print("Warning: NaN values detected in reconstruction errors.")
        return recon_errors, 0.0

    roc = roc_auc_score(y_test, recon_errors)
    return recon_errors, roc
