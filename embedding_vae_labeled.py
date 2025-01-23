import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labeled_embeddings = np.load("project/data/labeled_embeddings.npz")['embeddings']
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
    
print('Loading model')
model = torch.load("project/models/vae_emb_new_32d.pth").to(device)
print('Evaluating model')
model.eval()

# Get the latent space representation of the labeled embeddings

labeled_dataset = TensorDataset(torch.tensor(labeled_embeddings, dtype=torch.float32))

labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=False)

latent_space = []

with torch.no_grad():
    for data in labeled_loader:
        embeddings = data[0].to(device)
        mu, logvar = model.encode(embeddings)
        z = model.reparameterize(mu, logvar)
        latent_space.append(z)
    
latent_space = torch.cat(latent_space, dim=0).cpu()

print(f"Latent Space: {latent_space.shape}")

# Save the latent space representation to a npz file

np.savez("project/data/labeled_latent_space_32d.npz", latent_space=latent_space.numpy())

