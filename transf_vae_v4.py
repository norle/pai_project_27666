from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Define training parameters
num_epochs = 100
learning_rate = 1e-2
max_grad_norm = 100  # Maximum norm for gradient clipping

model_path = None
#model_path = 'project/models/transformer_vae_large.pth'
# Define the alphabet for protein sequences

data_path = "project/data/large_subunit_filtered.fasta"
protein_alphabet = "ACDEFGHIKLMNPQRSTVWY"

# Create a tokenizer with a BPE model
tokenizer = Tokenizer(models.BPE())

# Define a pre-tokenizer that splits on each character
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Define a decoder
tokenizer.decoder = decoders.ByteLevel()

# Define a trainer with the protein alphabet
trainer = trainers.BpeTrainer(
    vocab_size=len(protein_alphabet) + 3,  # +3 for <pad>, <eos>, and <s>
    special_tokens=["<pad>", "<eos>", "<s>"],
    initial_alphabet=list(protein_alphabet)
)

# Train the tokenizer on a list of protein sequences
protein_sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRSTVWY"]
tokenizer.train_from_iterator(protein_sequences, trainer=trainer)

# Add post-processing to handle special tokens
tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A <eos>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("<eos>", tokenizer.token_to_id("<eos>")),
    ],
)

# Save the tokenizer
tokenizer.save("protein_tokenizer.json")

# Example usage
encoded = tokenizer.encode("ACDEFGHIKLMNPQRSTVWY")
print(encoded.tokens)
vocab = tokenizer.get_vocab()
print("Vocabulary:", vocab)
print("Vocabulary size:", len(vocab))

#%% Load the tokenizer

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from Bio import SeqIO
import math
import matplotlib.pyplot as plt

BATCH_SIZE = 128

seq_length = 500

# Read a fasta file into a dataframe
def read_fasta_to_df(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append({"ID": record.id, "Sequence": str(record.seq)})
    return pd.DataFrame(sequences)



# Example usage
fasta_file = data_path
df = read_fasta_to_df(fasta_file)
print(df.head())

# Delete entries where df['Sequence'] is longer than max seq length
df = df[df['Sequence'].str.len() <= seq_length-1]
# Filter sequences longer than 20
#df = df[df['Sequence'].str.len() > 40]

# Tokenize the sequences
df['Tokenized Sequence'] = df['Sequence'].apply(lambda x: tokenizer.encode(x).tokens)

# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Convert tokenized sequences to numerical IDs
def tokenize_sequence(sequence):
    return [tokenizer.token_to_id(token) for token in sequence]

train_sequences = train_df['Tokenized Sequence'].apply(tokenize_sequence).tolist()
val_sequences = val_df['Tokenized Sequence'].apply(tokenize_sequence).tolist()
test_sequences = test_df['Tokenized Sequence'].apply(tokenize_sequence).tolist()

# Convert to PyTorch tensors
train_tensors = [torch.tensor(seq, dtype=torch.long) for seq in train_sequences]
val_tensors = [torch.tensor(seq, dtype=torch.long) for seq in val_sequences]
test_tensors = [torch.tensor(seq, dtype=torch.long) for seq in test_sequences]

# Apply padding to the sequences till length 600
train_tensors = torch.nn.utils.rnn.pad_sequence(train_tensors, batch_first=True, padding_value=tokenizer.token_to_id("<pad>"))
val_tensors = torch.nn.utils.rnn.pad_sequence(val_tensors, batch_first=True, padding_value=tokenizer.token_to_id("<pad>"))
test_tensors = torch.nn.utils.rnn.pad_sequence(test_tensors, batch_first=True, padding_value=tokenizer.token_to_id("<pad>"))

# Ensure all sequences are of length 600
train_tensors = torch.nn.functional.pad(train_tensors, (0, seq_length - train_tensors.size(1)), value=tokenizer.token_to_id("<pad>"))
val_tensors = torch.nn.functional.pad(val_tensors, (0, seq_length - val_tensors.size(1)), value=tokenizer.token_to_id("<pad>"))
test_tensors = torch.nn.functional.pad(test_tensors, (0, seq_length - test_tensors.size(1)), value=tokenizer.token_to_id("<pad>"))

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

class TransformerVAE(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, latent_dim, dropout=0):
        super(TransformerVAE, self).__init__()
        self.embedding = nn.Embedding(input_dim, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_mu = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, latent_dim)
        )
        
        self.fc_latent = nn.Sequential(
            nn.Linear(latent_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, model_dim)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(model_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def positional_encoding(self, seq_length, model_dim, device):
        position = torch.arange(0, seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(seq_length, model_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def encode(self, src):
        src_seq_length = src.size(1)
        src_pos = self.positional_encoding(src_seq_length, self.embedding.embedding_dim, src.device).unsqueeze(0)
        
        src = self.embedding(src) + src_pos
        src = self.dropout(src)
        
        memory = self.encoder(src)
        memory = memory.mean(dim=1)
        
        mu = self.fc_mu(memory)
        logvar = self.fc_logvar(memory)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, tgt):
        tgt_seq_length = tgt.size(1)
        tgt_pos = self.positional_encoding(tgt_seq_length, self.embedding.embedding_dim, tgt.device).unsqueeze(0)
        
        tgt = self.embedding(tgt) + tgt_pos
        tgt = self.dropout(tgt)
        
        z = self.fc_latent(z).unsqueeze(1).repeat(1, tgt_seq_length, 1)
        output = self.decoder(tgt, z)
        output = output[:, -1:, :]
        output = self.fc_out(output)
        
        return output

    def forward(self, src, tgt):
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, tgt)
        return output, mu, logvar

# Define model parameters
input_dim = len(vocab)
model_dim = 64
num_heads = 8
num_layers = 4
output_dim = len(vocab)
latent_dim = 16

# Instantiate the model
if model_path != None:
    model = torch.load(model_path).to(device)
else:
    model = TransformerVAE(input_dim, model_dim, num_heads, num_layers, output_dim, latent_dim).to(device)

# Define loss function and optimizer
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

#criterion = FocalLoss(ignore_index=tokenizer.token_to_id("<pad>"))
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"), reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.8)

val_losses = []
train_losses = []
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    loss_counter = 0
    for nx, batch in enumerate(train_loader):

        src = batch[0][:, :].to(device)
        tgt = batch[0][:, :].to(device)
        
        
        
        # Gradually mask the source
        for i in range(1, src.size(1)):
            optimizer.zero_grad()
            masked_src = src[:, :i] # source with tokens masked after position i
            tgt = src[:, i] # we predict ith token (one after mask)
            if tgt.eq(0).all():
                continue

            output, mu, logvar = model(src, masked_src)
            recon_loss = criterion(output.reshape(-1, output_dim), tgt.reshape(-1))
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.1 * kld_loss
            
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            total_loss += loss.item()
            loss_counter += 1
        print(f"Batch {nx} loss: {total_loss / loss_counter:.4f}")
    
    avg_loss = total_loss / loss_counter
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Validation loop
    model.eval()
    val_loss = 0
    val_loss_counter = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch[0][:, :].to(device)
            tgt = batch[0][:, :].to(device)
            
            for i in range(1, src.size(1)):
                masked_src = src[:, :i]
                tgt = src[:, i]
                if tgt.eq(0).all():
                    continue
                
                output_val, mu, logvar = model(src, masked_src)
                
                recon_loss = criterion(output_val.reshape(-1, output_dim), tgt.reshape(-1))
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld_loss
                
                val_loss += loss.item()
                val_loss_counter += 1

    avg_val_loss = val_loss / val_loss_counter
    val_losses.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    torch.save(model, "project/models/transformer_vae.pth")
    val_loss = 0
    val_loss_counter = 0
    with torch.no_grad():
        for batch in val_loader:
            src = batch[0][:, :].to(device)
            tgt = batch[0][:, :].to(device)
            
            for i in range(1, src.size(1)):
                masked_src = src[:, :i]
                tgt = src[:, i]
                if tgt.eq(0).all():
                    continue
                
                output_val, mu, logvar = model(src, masked_src)
                
                recon_loss = criterion(output_val.reshape(-1, output_dim), tgt.reshape(-1))
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld_loss
                
                val_loss += loss.item()
                val_loss_counter += 1

    avg_val_loss = val_loss / val_loss_counter
    val_losses.append(avg_val_loss)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    torch.save(model, "project/models/transformer_vae_large.pth")

#%% Save the model

torch.save(model, "project/models/transformer_vae_1.pth")

#%% # Visualize the losses
import matplotlib.pyplot as plt

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('transformer_vae_loss.png')

#%% Generate sequences
#model = TransformerVAE(input_dim, model_dim, num_heads, num_layers, output_dim, latent_dim).to(device)
with torch.no_grad():
    for batch in test_loader:
        src = batch[0][:, :].to(device)
        
        # Initialize gen_seq with the same batch size as src
        gen_seq = torch.tensor([[3]] * src.size(0)).to(device)
        
        for i in range(1, src.size(1)):
            masked_src = src[:, :i]
            tgt = src[:, i]
            if tgt.eq(0).all():
                continue
            
            output_val, mu, logvar = model(src, gen_seq)
            next_token = torch.argmax(output_val, dim=-1)
            if next_token.ndim == 1:
                next_token = next_token.unsqueeze(1)
            gen_seq = torch.cat([gen_seq, next_token[:, -1].unsqueeze(1)], dim=-1)

        print("Generated Sequence:", gen_seq[0])
        print("Actual Sequence:", src[0])
