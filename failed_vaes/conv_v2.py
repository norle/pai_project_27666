from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Define training parameters
num_epochs = 100
learning_rate = 1e-4
max_grad_norm = 100  # Maximum norm for gradient clipping

model_path = None
#model_path = 'project/models/transformer_vae_large.pth'
# Define the alphabet for protein sequences

#data_path = "project/data/interpro_sequences_all_17_01.fasta"
#data_path = "project/data/large_subunit_filtered.fasta"
data_path = 'project/data/clustered90_seq_rep_seq.fasta'
protein_alphabet = "ACDEFGHIKLMNPQRSTVWY-"




#%% Load the tokenizer

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from Bio import SeqIO
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 32

seq_length = 160

# Read a fasta file into a dataframe
def read_fasta_to_df(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append({"ID": record.id, "Sequence": str(record.seq)})
    return pd.DataFrame(sequences)

def one_hot_encode_sequence(seq, alphabet, max_len):
    import torch
    one_hot = torch.zeros(len(alphabet), max_len)
    for i, aa in enumerate(seq[:max_len]):
        if aa in alphabet:
            idx = alphabet.index(aa)
            one_hot[idx, i] = 1.0
    for i in range(len(seq), max_len):
        one_hot[-1, i] = 1.0
    return one_hot

# Example usage
fasta_file = data_path
df = read_fasta_to_df(fasta_file)
print(df.head())

# Delete entries where df['Sequence'] is longer than max seq length
df = df[df['Sequence'].str.len() <= seq_length-1]
# Filter sequences longer than 20
df = df[df['Sequence'].str.len() > 20]

# Remove duplicate sequences
df = df.drop_duplicates(subset=['Sequence'])

# Remove references to the tokenizer and replace with one-hot encoding:
df['Encoded Sequence'] = df['Sequence'].apply(lambda x: one_hot_encode_sequence(x, protein_alphabet, seq_length))

# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_sequences = train_df['Encoded Sequence'].tolist()
val_sequences = val_df['Encoded Sequence'].tolist()
test_sequences = test_df['Encoded Sequence'].tolist()

# Convert to PyTorch tensors
train_tensors = torch.stack(train_sequences)
val_tensors = torch.stack(val_sequences)
test_tensors = torch.stack(test_sequences)

# Create datasets
train_dataset = TensorDataset(train_tensors)
val_dataset = TensorDataset(val_tensors)
test_dataset = TensorDataset(test_tensors)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

#%% Define and train the VAE model
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNVAE(nn.Module):
    def __init__(self, vocab_size, latent_dim, seq_length):
        super(CNNVAE, self).__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=21, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128 * (seq_length // 4), latent_dim)
        self.fc_logvar = nn.Linear(128 * (seq_length // 4), latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128 * (seq_length // 8))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=vocab_size, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, self.seq_length // 8)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

class ZeroBaseline(nn.Module):
    def forward(self, x):
        out = torch.zeros_like(x)
        mu = torch.zeros(x.size(0), latent_dim).to(x.device)
        logvar = torch.zeros(x.size(0), latent_dim).to(x.device)
        return out, mu, logvar
    
latent_dim = 64
vocab_size = len(protein_alphabet)

# Instantiate the model
if model_path != None:
    model = torch.load(model_path).to(device)
else:
    model = CNNVAE(vocab_size, latent_dim, seq_length).to(device)



criterion = nn.HuberLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Zero baseline prediction
baseline_model = ZeroBaseline().to(device)
baseline_model.eval()

baseline_train_loss = 0
with torch.no_grad():
    for batch in train_loader:
        src = batch[0].to(device)
        zero_pred, _, _ = baseline_model(src)
        baseline_train_loss += criterion(zero_pred, src).item()
baseline_train_loss /= len(train_loader)
print(f"Zero Baseline Train Loss: {baseline_train_loss:.4f}")

baseline_val_loss = 0
with torch.no_grad():
    for batch in val_loader:
        src = batch[0].to(device)
        zero_pred, _, _ = baseline_model(src)
        baseline_val_loss += criterion(zero_pred, src).item()
baseline_val_loss /= len(val_loader)
print(f"Zero Baseline Val Loss: {baseline_val_loss:.4f}")

val_losses = []
train_losses = []

warmup_epochs = 5

for epoch in range(num_epochs):
    if epoch < warmup_epochs:
        lr_this_epoch = learning_rate * float(epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_epoch
    else:
        scheduler.step()

    model.train()
    train_loss = 0
    for batch in train_loader:
        src = batch[0].to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(src)
        loss_recon = criterion(output, src)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = 1000*(loss_recon + 1e-1 * kld_loss)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        train_loss += total_loss.item()



    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch[0].to(device)
            output, mu, logvar = model(src)
            loss_recon = criterion(output, src)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            val_loss += (loss_recon + kld_loss).item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

#%% Save the model

torch.save(model, "project/models/transformer_vae_unfiltered.pth")

#%% # Visualize the losses
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('transformer_vae_loss.png')

#%%
# Visualize the predictions
import numpy as np

model.eval()
with torch.no_grad():
    for batch in test_loader:
        src = batch[0].to(device)
        output, _, _ = model(src)
        break

output = output.cpu().numpy()
src = src.cpu().numpy()

def one_hot_decode_sequence(seq, alphabet):
    decoded = ""
    for i in range(seq.shape[1]):
        idx = np.argmax(seq[:, i])
        decoded += alphabet[idx]
    return decoded

print("Original Sequence:")
decoded_src = [one_hot_decode_sequence(s, protein_alphabet) for s in src[:3]]
decoded_output = [one_hot_decode_sequence(o, protein_alphabet) for o in output[:3]]
for i in range(3):
    print("Original:", decoded_src[i])
    print("Reconstructed:", decoded_output[i])
    print("-----")


