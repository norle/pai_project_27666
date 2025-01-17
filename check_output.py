from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Define training parameters
num_epochs = 100
learning_rate = 1e-3
max_grad_norm = 10  # Maximum norm for gradient clipping

model_path = None
model_path = 'project/models/transformer_vae.pth'
# Define the alphabet for protein sequences
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
    special_tokens=["<pad>", "<mask>", "<eos>", "<s>"],
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

BATCH_SIZE = 32

seq_length = 200

# Read a fasta file into a dataframe
def read_fasta_to_df(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append({"ID": record.id, "Sequence": str(record.seq)})
    return pd.DataFrame(sequences)



# Example usage
fasta_file = "project/data/clustered90_seq_rep_seq.fasta"
df = read_fasta_to_df(fasta_file)
print(df.head())

# Delete entries where df['Sequence'] is longer than max seq length
df = df[df['Sequence'].str.len() <= seq_length-1]

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
        self.pos_encoder = nn.Embedding(seq_length, model_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_mu = nn.Linear(model_dim, latent_dim)
        self.fc_logvar = nn.Linear(model_dim, latent_dim)
        
        self.fc_latent = nn.Linear(latent_dim, model_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Sequential(
            nn.Linear(model_dim, output_dim*2),
            nn.ReLU(),
            nn.Linear(output_dim*2, output_dim),

            )
        self.dropout = nn.Dropout(dropout)

    def encode(self, src):
        src_seq_length = src.size(1)
        src_pos = torch.arange(0, src_seq_length, device=src.device).unsqueeze(0)
        
        src = self.embedding(src) + self.pos_encoder(src_pos)
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
        tgt_pos = torch.arange(0, tgt_seq_length, device=tgt.device).unsqueeze(0)
        
        tgt = self.embedding(tgt) + self.pos_encoder(tgt_pos)
        tgt = self.dropout(tgt)
        
        z = self.fc_latent(z).unsqueeze(1).repeat(1, tgt_seq_length, 1)
        
        output = self.decoder(tgt, z)
        output = output[:, -1:, :]
        #output = output.mean(dim=1)
        #print(output.shape)
        output = self.fc_out(output)
        
        # Ensure the output shape is (32, 1, 24)
        #print(output.shape)
        
        
        
        
        return output

    def forward(self, src, tgt):
        mu, logvar = self.encode(src)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z, tgt)
        return output, mu, logvar

# Define model parameters
input_dim = len(vocab)
model_dim = 112
num_heads = 4
num_layers = 3
output_dim = len(vocab)
latent_dim = 24

# Instantiate the model
if model_path != None:
    model = torch.load(model_path).to(device)
else:
    model = TransformerVAE(input_dim, model_dim, num_heads, num_layers, output_dim, latent_dim).to(device)



# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"), reduction='mean')


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

        # Collect all latent vectors
        latent_vectors = []
        with torch.no_grad():
            for batch in test_loader:
                src = batch[0][:, :].to(device)
                mu, logvar = model.encode(src)
                z = model.reparameterize(mu, logvar)
                latent_vectors.append(z.cpu().numpy())

        latent_vectors = np.concatenate(latent_vectors, axis=0)

        # Apply PCA to reduce the dimensionality of the latent space
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_vectors)

        # Plot the latent space
        plt.figure(figsize=(8, 6))
        plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.5)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Latent Space')
        plt.show()