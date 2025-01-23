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

data_path = "project/data/labeled_data_sorted.csv"

# Read a CSV file into a dataframe
def read_csv_to_df(csv_file):
    return pd.read_csv(csv_file)

seq_length = 500

# Example usage
csv_file = data_path
df = read_csv_to_df(csv_file)
df = df[df["sequence"].str.len() <= seq_length]
print(df.head())

# Example usage with sequences from the dataframe
sequences = df["sequence"].tolist()

inputs = tokenizer(sequences, return_tensors="pt", padding='max_length', truncation=True, max_length=seq_length)

dataset = [{"input_ids": input_id, "attention_mask": attention_mask, "labels": input_id} for input_id, attention_mask in zip(inputs["input_ids"], inputs["attention_mask"])]

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

# Get embeddings for the dataset
embeddings = np.array(get_embeddings(dataset, model_encoder, tokenizer, device))

embeddings = embeddings.mean(axis=1)

# Save embeddings to npz file
np.savez("project/data/labeled_embeddings.npz", embeddings=embeddings)

print(f"Embeddings: {embeddings.shape}")

