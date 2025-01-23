from transformers import AutoTokenizer, AutoModel
from transformers import AutoConfig
import transformers
from transformers import EncoderDecoderModel
from transformers import Trainer, TrainingArguments
import pandas as pd
from Bio import SeqIO
from sklearn.model_selection import train_test_split
import random
import string
import torch
import numpy as np


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer and model
model_checkpoint = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model_encoder = AutoModel.from_pretrained(model_checkpoint).to(device)



data_path = "project/data/large_subunit_filtered.fasta"
#data_path = 'project/data/clustered90_seq_rep_seq.fasta'


# Read a fasta file into a dataframe
def read_fasta_to_df(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append({"ID": record.id, "Sequence": str(record.seq)})
    return pd.DataFrame(sequences)

seq_length = 500

# Example usage
fasta_file = data_path
df = read_fasta_to_df(fasta_file)
df = df[df["Sequence"].str.len() <= seq_length]
print(df.head())
# Split the dataframe into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
# Example usage with sequences from the dataframe
train_sequences = train_df["Sequence"].tolist()
val_sequences = val_df["Sequence"].tolist()
test_sequences = test_df["Sequence"].tolist()

train_inputs = tokenizer(train_sequences, return_tensors="pt", padding='max_length', truncation=True, max_length=seq_length)
val_inputs = tokenizer(val_sequences, return_tensors="pt", padding='max_length', truncation=True, max_length=seq_length)
test_inputs = tokenizer(test_sequences, return_tensors="pt", padding='max_length', truncation=True, max_length=seq_length)

train_dataset = [{"input_ids": input_id, "attention_mask": attention_mask, "labels": input_id} for input_id, attention_mask in zip(train_inputs["input_ids"], train_inputs["attention_mask"])]
val_dataset = [{"input_ids": input_id, "attention_mask": attention_mask, "labels": input_id} for input_id, attention_mask in zip(val_inputs["input_ids"], val_inputs["attention_mask"])]
test_dataset = [{"input_ids": input_id, "attention_mask": attention_mask, "labels": input_id} for input_id, attention_mask in zip(test_inputs["input_ids"], test_inputs["attention_mask"])]

# Function to get embeddings from the encoder
def get_embeddings(dataset, model, tokenizer, device):
    embeddings = []
    for data in dataset:
        input_ids = data["input_ids"].unsqueeze(0).to(device)
        attention_mask = data["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        embeddings.append(outputs.last_hidden_state.squeeze().cpu().numpy())
    return embeddings

# Get embeddings for train, validation, and test datasets
train_embeddings = np.array(get_embeddings(train_dataset, model_encoder, tokenizer, device))
val_embeddings = np.array(get_embeddings(val_dataset, model_encoder, tokenizer, device))
test_embeddings = np.array(get_embeddings(test_dataset, model_encoder, tokenizer, device))

train_embeddings = train_embeddings.mean(axis=1)
val_embeddings = val_embeddings.mean(axis=1)
test_embeddings = test_embeddings.mean(axis=1)

# Save embeddings to npz files
np.savez("project/data/train_embeddings.npz", embeddings=train_embeddings)
np.savez("project/data/val_embeddings.npz", embeddings=val_embeddings)
np.savez("project/data/test_embeddings.npz", embeddings=test_embeddings)

print(f"Train Embeddings: {train_embeddings.shape}")
print(f"Validation Embeddings: {val_embeddings.shape}")
print(f"Test Embeddings: {test_embeddings.shape}")
#print(outputs)

