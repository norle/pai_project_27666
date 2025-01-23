import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

# Load data
labeled_data = pd.read_csv("project/data/labeled_data_sorted.csv")
latent_space_8d = np.load("project/data/labeled_latent_space_8d.npz")['latent_space']
latent_space_2d = np.load("project/data/labeled_latent_space_2d.npz")['latent_space']
labeled_emb = np.load("project/data/labeled_embeddings.npz")['embeddings']

labeled_data.reset_index(inplace=True)
labeled_data['latent_space_8d'] = latent_space_8d.tolist()
labeled_data['latent_space_2d'] = latent_space_2d.tolist()
labeled_data['embeddings'] = labeled_emb.tolist()

# Perform UMAP dimensionality reduction for 8D latent space
reducer_8d = umap.UMAP()
embedding_8d = reducer_8d.fit_transform(latent_space_8d)

# Perform UMAP dimensionality reduction for 2D latent space
reducer_2d = umap.UMAP()
embedding_2d = reducer_2d.fit_transform(latent_space_2d)

# Perform UMAP dimensionality reduction for embeddings
reducer_emb = umap.UMAP()
embedding_emb = reducer_emb.fit_transform(labeled_emb)

# Plot the UMAP result for 8D latent space
plt.figure(figsize=(10, 8))
scatter_8d = plt.scatter(embedding_8d[:, 0], embedding_8d[:, 1], c=labeled_data['Zero-shot score'], cmap='viridis', s=5)
plt.colorbar(scatter_8d, label='Zero-shot score')
plt.title('UMAP projection of the 8D latent space')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# Plot the UMAP result for 2D latent space
plt.figure(figsize=(10, 8))
scatter_2d = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=labeled_data['Zero-shot score'], cmap='viridis', s=5)
plt.colorbar(scatter_2d, label='Zero-shot score')
plt.title('UMAP projection of the 2D latent space')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

# Plot the UMAP result for embeddings
plt.figure(figsize=(10, 8))
scatter_emb = plt.scatter(embedding_emb[:, 0], embedding_emb[:, 1], c=labeled_data['Zero-shot score'], cmap='viridis', s=5)
plt.colorbar(scatter_emb, label='Zero-shot score')
plt.title('UMAP projection of the embeddings')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()