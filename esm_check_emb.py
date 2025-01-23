import numpy as np

train_embeddings = np.load("project/data/train_embeddings.npz")['embeddings']
val_embeddings = np.load("project/data/val_embeddings.npz")['embeddings']
test_embeddings = np.load("project/data/test_embeddings.npz")['embeddings']

print(train_embeddings)

print(f"Train Embeddings: {len(train_embeddings)}, Shape: {train_embeddings.shape}")
print(f"Validation Embeddings: {len(val_embeddings)}, Shape: {val_embeddings.shape}")
print(f"Test Embeddings: {len(test_embeddings)}, Shape: {test_embeddings.shape}")
print('Embedding size:', train_embeddings[0].shape)

print("Train Embeddings Statistics:")
print(f"Mean: {np.mean(train_embeddings, axis=0)}")
print(f"Standard Deviation: {np.std(train_embeddings, axis=0)}")
print(f"Min: {np.min(train_embeddings, axis=0)}")
print(f"Max: {np.max(train_embeddings, axis=0)}")

print("Validation Embeddings Statistics:")
print(f"Mean: {np.mean(val_embeddings, axis=0)}")
print(f"Standard Deviation: {np.std(val_embeddings, axis=0)}")
print(f"Min: {np.min(val_embeddings, axis=0)}")
print(f"Max: {np.max(val_embeddings, axis=0)}")

print("Test Embeddings Statistics:")
print(f"Mean: {np.mean(test_embeddings, axis=0)}")
print(f"Standard Deviation: {np.std(test_embeddings, axis=0)}")
print(f"Min: {np.min(test_embeddings, axis=0)}")
print(f"Max: {np.max(test_embeddings, axis=0)}")