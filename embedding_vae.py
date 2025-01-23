import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA

BATCH_SIZE = 64
LR = 1e-4
LATENT_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 2000

train_embeddings = np.load("project/data/train_embeddings.npz")['embeddings']
val_embeddings = np.load("project/data/val_embeddings.npz")['embeddings']
test_embeddings = np.load("project/data/test_embeddings.npz")['embeddings']

train_dataset = TensorDataset(torch.tensor(train_embeddings, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(val_embeddings, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(test_embeddings, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train Embeddings: {len(train_embeddings)}, Shape: {train_embeddings.shape}")
print(f"Validation Embeddings: {len(val_embeddings)}, Shape: {val_embeddings.shape}")
print(f"Test Embeddings: {len(test_embeddings)}, Shape: {test_embeddings.shape}")

import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=320, hidden_dim=128, latent_dim=64):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log variance
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Check if GPU is available and select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the VAE
vae = VAE(input_dim=train_embeddings.shape[1], hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)

# Print the model architecture
print(vae)

# Loss function
def loss_function(recon_x, x, mu, logvar, kl_weight):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl_constant_weight = 1e-4
    return MSE + 1e-3 * kl_weight * KLD

# Optimizer
optimizer = torch.optim.Adam(vae.parameters(), lr=LR)

# KL annealing function
def kl_annealing(epoch, cycle_length=200):
    return (epoch % cycle_length) / cycle_length

# Training loop
num_epochs = EPOCHS
for epoch in range(num_epochs):
    vae.train()
    train_loss = 0
    kl_weight = kl_annealing(epoch)
    for batch in train_loader:
        data = batch[0].to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar, kl_weight)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    # Validation
    vae.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            data = batch[0].to(device)
            recon_batch, mu, logvar = vae(data)
            loss = loss_function(recon_batch, data, mu, logvar, kl_weight)
            val_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader.dataset)}, Validation Loss: {val_loss / len(val_loader.dataset)}')

import matplotlib.pyplot as plt

# Visualize the latent space
vae.eval()
with torch.no_grad():
    z_test = []
    for batch in test_loader:
        data = batch[0].to(device)
        mu, logvar = vae.encode(data)
        z = vae.reparameterize(mu, logvar)
        z_test.append(z.cpu().numpy())
    z_test = np.concatenate(z_test, axis=0)

# Calculate test loss
test_loss = 0
vae.eval()
with torch.no_grad():
    for batch in test_loader:
        data = batch[0].to(device)
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar, kl_weight)
        test_loss += loss.item()

print(f'Test Loss: {test_loss / len(test_loader.dataset)}')

plt.figure(figsize=(8, 6))
plt.scatter(z_test[:, 0], z_test[:, 1], s=2)
plt.title('Latent Space Visualization of Test Data')
plt.xlabel('z1')
plt.ylabel('z2')
plt.savefig('project/figures/latent_space_emb.png')

# Function to plot PCA
def plot_pca(data, title, filename):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=2)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(filename)

# Plot PCA for input representations
plot_pca(train_embeddings, 'PCA of Input Representations', 'project/figures/pca_input_emb.png')

# Plot PCA for output representations
vae.eval()
with torch.no_grad():
    reconstructions = []
    for batch in train_loader:
        data = batch[0].to(device)
        recon_batch, _, _ = vae(data)
        reconstructions.append(recon_batch.cpu().numpy())
    reconstructions = np.concatenate(reconstructions, axis=0)
print(reconstructions.shape)

plot_pca(reconstructions, 'PCA of Output Representations', 'project/figures/pca_output_emb.png')

# Save the model
torch.save(vae, 'project/models/vae_emb_new_32d.pth')
