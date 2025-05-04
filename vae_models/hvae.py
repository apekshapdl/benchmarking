# Hierarchical VAE in PyTorch
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import numpy as np

# ----------------------
# Reparameterization Trick
# ----------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    std = torch.clamp(std, min=1e-6)  
    eps = torch.randn_like(std)
    return mu + eps * std

# ----------------------
# Encoder Networks
# ----------------------
class EncoderZ2(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.dropout = nn.Dropout(p=0.2)  # Dropout probability can be tuned

    def forward(self, x):
        h = self.dropout(F.relu(self.fc1(x)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class EncoderZ1(nn.Module):
    def __init__(self, input_dim, latent_dim, z2_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + z2_dim, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.dropout = nn.Dropout(p=0.2)  # Dropout probability can be tuned

    def forward(self, x, z2):
        h = torch.cat([x, z2], dim=-1)
        h = self.dropout(F.relu(self.fc1(h)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# ----------------------
# Decoder Network
# ----------------------
class Decoder(nn.Module):
    def __init__(self, z1_dim, z2_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(z1_dim + z2_dim, 256)
        self.fc_out = nn.Linear(256, output_dim)
        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, z1, z2):
        h = torch.cat([z1, z2], dim=-1)
        h = F.relu(self.fc1(h))
        out = self.fc_out(h)
        out = torch.clamp(out, min=-10, max=10)
        return self.fc_out(h)  # Remove sigmoid for real-valued outputs

# ----------------------
# Hierarchical VAE Model
# ----------------------
class HierarchicalVAE(nn.Module):
    def __init__(self, input_dim, z1_dim, z2_dim):
        super().__init__()
        self.encoder_z2 = EncoderZ2(input_dim, z2_dim)
        self.encoder_z1 = EncoderZ1(input_dim, z1_dim, z2_dim)
        self.decoder = Decoder(z1_dim, z2_dim, input_dim)
        

    def forward(self, x):
        mu_z2, logvar_z2 = self.encoder_z2(x)
        z2 = reparameterize(mu_z2, logvar_z2)

        mu_z1, logvar_z1 = self.encoder_z1(x, z2)
        z1 = reparameterize(mu_z1, logvar_z1)

        x_hat = self.decoder(z1, z2)
        return x_hat, mu_z1, logvar_z1, mu_z2, logvar_z2

# ----------------------
# Loss Function (Hierarchical ELBO)
# ----------------------
def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def loss_function(x, x_hat, mu_z1, logvar_z1, mu_z2, logvar_z2, kl_weight=1.0):
    recon_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
    kl_z1 = kl_divergence(mu_z1, logvar_z1)
    kl_z2 = kl_divergence(mu_z2, logvar_z2)
    total_loss = recon_loss + kl_weight * (kl_z1 + kl_z2)
    return total_loss, recon_loss, kl_z1, kl_z2

def evaluate_model(model, X_test_tensor, y_test, batch_size=128):
    model.eval()
    recon_errors = []

    with torch.no_grad():
        loader = DataLoader(TensorDataset(X_test_tensor), batch_size=batch_size)
        for batch in loader:
            x = batch[0]

            x = torch.clamp(x, min=-10, max=10)

            x_hat, _, _, _, _ = model(x)
            x_hat = torch.clamp(x_hat, min=-10, max=10)
            errors = torch.mean((x - x_hat) ** 2, dim=1).cpu().numpy()
            recon_errors.extend(errors)

    recon_errors = np.array(recon_errors)
    recon_errors = (recon_errors - recon_errors.min()) / (recon_errors.max() - recon_errors.min() + 1e-8)  # numerical stability

    if np.isnan(recon_errors).any():
        print("Warning: NaN values detected in reconstruction errors. Skipping AUC.")
        return recon_errors, 0.0

    if len(recon_errors) != len(y_test):
        print("Warning: Length mismatch between recon_errors and y_test.")
        return recon_errors, 0.0

    auc = roc_auc_score(y_test, recon_errors)
    return recon_errors, auc



def train_models(X_train, X_test, y_test, input_dim, z1_list, z2_list, epoch_list, optimizer_type="adam"):

    # Normalize data
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    eps = 1e-8  
    std_adj = np.where(std == 0, eps, std)

    X_train = (X_train - mean) / std_adj
    X_test = (X_test - mean) / std_adj


    #convert numpy arrays into pytorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test = y_test.astype(int)

    # DataLoader factory
    train_loader = lambda: DataLoader(TensorDataset(X_train_tensor), batch_size=64, shuffle=True)

    results = []

    for z1_dim in z1_list:
        for z2_dim in z2_list:
            for epochs in epoch_list:
                model = HierarchicalVAE(input_dim, z1_dim, z2_dim)

                # Choose optimizer
                if optimizer_type.lower() == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                elif optimizer_type.lower() == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                else:
                    raise ValueError("Unsupported optimizer: choose 'adam' or 'sgd'.")

                # Train loop
                model.train()
                for epoch in range(epochs):
                    kl_weight = min(1.0, epoch / 50)  # Linear annealing for first 50 epochs
                    for (x_batch,) in train_loader():
                        x_hat, mu_z1, logvar_z1, mu_z2, logvar_z2 = model(x_batch)
                        loss, _, _, _ = loss_function(x_batch, x_hat, mu_z1, logvar_z1, mu_z2, logvar_z2, kl_weight=kl_weight)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Evaluate
                recon_errors, roc_auc = evaluate_model(model, X_test_tensor, y_test)
                precision, recall, _ = precision_recall_curve(y_test, recon_errors)
                pr_auc = auc(recall, precision)


                results.append({
                    "z1_dim": z1_dim,
                    "z2_dim": z2_dim,
                    "epochs": epochs,
                    "optimizer": optimizer_type,
                    "roc_auc": roc_auc,
                    "pr_auc" : pr_auc,
                    "model": model  # return the trained model
})

                print(f"z1: {z1_dim}, z2: {z2_dim}, epochs: {epochs}, opt: {optimizer_type.upper()} → ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    return results