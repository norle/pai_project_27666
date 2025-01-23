from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Define training parameters
num_epochs = 200
learning_rate = 1e-3
max_grad_norm = 100  # Maximum norm for gradient clipping
kld_weight = 0.1  # Initial weight for KLD loss
kld_max_weight = 1.0  # Maximum weight for KLD loss
kld_cycle_length = 50  # Number of epochs per cycle

model_path = None
#model_path = 'project/models/transformer_vae_large.pth'
# Define the alphabet for protein sequences

#data_path = "project/data/interpro_sequences_all_17_01.fasta"
data_path = "project/data/large_subunit_filtered.fasta"
#data_path = 'project/data/clustered90_seq_rep_seq.fasta'
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
df = df[df['Sequence'].str.len() > 20]

# Remove duplicate sequences
df = df.drop_duplicates(subset=['Sequence'])

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerAutoregressor(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_len=5000, latent_dim=32):
        super(TransformerAutoregressor, self).__init__()
        self.shared_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.decoder_norm = nn.LayerNorm(d_model)

        self.fc_mu = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, latent_dim)
        )
        self.fc_latent_to_d_model = nn.Sequential(
            nn.Linear(latent_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

        self.conv_encoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )
        self.conv_decoder = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        )

    def encode(self, src, src_mask=None, src_padding_mask=None):
        src = self.shared_embedding(src) * (self.shared_embedding.embedding_dim ** 0.5)
        src = src.transpose(0, 1)
        src = self.positional_encoding(src)
        memory = self.encoder(src, src_mask, src_padding_mask)
        memory = self.encoder_norm(memory)
        memory = memory.transpose(0, 1).transpose(1, 2)
        memory = self.conv_encoder(memory)
        memory = memory.transpose(1, 2)
        memory = memory.mean(dim=1)
        mu = self.fc_mu(memory)
        logvar = self.fc_logvar(memory)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, trg, trg_mask=None, trg_padding_mask=None, memory_key_padding_mask=None):
        memory = self.fc_latent_to_d_model(z).unsqueeze(0).repeat(trg.size(1), 1, 1)
        memory = memory.transpose(0, 1).transpose(1, 2)
        memory = self.conv_decoder(memory)
        memory = memory.transpose(1, 2).transpose(0, 1)
        trg = self.shared_embedding(trg) * (self.shared_embedding.embedding_dim ** 0.5)
        trg = trg.transpose(0, 1)
        trg = self.positional_encoding(trg)
        output = self.decoder(trg, memory, trg_mask, None, trg_padding_mask, memory_key_padding_mask)
        output = self.decoder_norm(output)
        return self.output_layer(output)

    def forward(self, src, trg, src_mask=None, trg_mask=None, src_padding_mask=None, trg_padding_mask=None, memory_key_padding_mask=None):
        mu, logvar = self.encode(src, src_mask, src_padding_mask)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, trg, trg_mask, trg_padding_mask, memory_key_padding_mask)
        return output, mu, logvar

    def generate_square_subsequent_mask(self, size):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
        return mask

    
vocab_size = len(vocab)
model_dim = 200
num_heads = 4
num_layers = 3
output_dim = len(vocab)
latent_dim = 32

# Instantiate the model
if model_path != None:
    model = torch.load(model_path).to(device)
else:
    model = TransformerAutoregressor(vocab_size, model_dim, num_heads, num_layers, dim_feedforward=1024, dropout=0.05, max_len=seq_length, latent_dim=latent_dim).to(device)



criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"), reduction='mean')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

val_losses = []
train_losses = []

warmup_epochs = 5

for epoch in range(num_epochs):
    # KLD weight cycling
    cycle_position = (epoch % kld_cycle_length) / kld_cycle_length
    kld_weight = kld_max_weight * (1 - abs(2 * cycle_position - 1))

    if epoch < warmup_epochs:
        lr_this_epoch = 1e-10 + (learning_rate - 1e-10) * (epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_epoch
    else:
        scheduler.step()

    model.train()
    train_loss = 0
    for batch in train_loader:
        src = batch[0].to(device)
        trg = batch[0].to(device)  # Using the same sequence as target for reconstruction

        optimizer.zero_grad()
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]

        src_mask = model.generate_square_subsequent_mask(src.size(1)).to(device)
        trg_mask = model.generate_square_subsequent_mask(trg_input.size(1)).to(device)

        output, mu, logvar = model(src, trg_input, src_mask=src_mask, trg_mask=trg_mask)
        output = output.view(-1, output.size(-1))
        trg_output = trg_output.contiguous().view(-1)

        loss = criterion(output, trg_output)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = loss + 1e-3* kld_weight * kld_loss

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
            trg = batch[0].to(device)

            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]

            src_mask = model.generate_square_subsequent_mask(src.size(1)).to(device)
            trg_mask = model.generate_square_subsequent_mask(trg_input.size(1)).to(device)

            output, mu, logvar = model(src, trg_input, src_mask=src_mask, trg_mask=trg_mask)
            output = output.view(-1, output.size(-1))
            trg_output = trg_output.contiguous().view(-1)

            loss = criterion(output, trg_output)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_loss = loss + kld_weight * kld_loss

            val_loss += total_loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, KLD Weight: {kld_weight:.4f}")

    if epoch % 5 == 0:
        torch.save(model, "project/models/transformer_vae_unfiltered.pth")
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

#%% Generate sequences
#model = TransformerVAE(input_dim, model_dim, num_heads, num_layers, output_dim, latent_dim).to(device)
with torch.no_grad():
    for batch in test_loader:
        src = batch[0].to(device)
        mu, logvar = model.encode(src)
        z = model.reparameterize(mu, logvar)

        # Initialize generation with <s> token
        start_id = tokenizer.token_to_id("<s>")
        gen_seq = torch.full((src.size(0), 1), start_id, dtype=torch.long).to(device)

        # Autoregressive decoding
        for _ in range(seq_length):
            output = model.decode(z, gen_seq)
            next_token = torch.argmax(output[-1], dim=-1).unsqueeze(1)
            gen_seq = torch.cat([gen_seq, next_token], dim=1)

        print("Generated Sequence:", gen_seq[0])
        print("Actual Sequence:", src[0])
