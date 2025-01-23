import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

labeled_data = pd.read_csv("project/data/labeled_data_sorted.csv")
latent_space_8d = np.load("project/data/labeled_latent_space_8d.npz")['latent_space']
latent_space_2d = np.load("project/data/labeled_latent_space_2d.npz")['latent_space']
labeled_emb = np.load("project/data/labeled_embeddings.npz")['embeddings']
latent_space_32d = np.load("project/data/labeled_latent_space_32d.npz")['latent_space']

labeled_data.reset_index(inplace=True)
labeled_data['latent_space_8d'] = latent_space_8d.tolist()
labeled_data['latent_space_2d'] = latent_space_2d.tolist()
labeled_data['embeddings'] = labeled_emb.tolist()
labeled_data['latent_space_32d'] = latent_space_32d.tolist()

labeled_data['rank'] = labeled_data['Zero-Shot Score'].rank(ascending=False)

def run_bo(df, rand_state=42):

    nr_of_samples = df.shape[0]
    print(f"Number of samples: {nr_of_samples}")
    
    rng = np.random.RandomState(rand_state)
    start_ix = rng.choice(int(0.5 * nr_of_samples), 10, replace=False)

    train_set = df[df['rank'].isin(start_ix)]

    train_flag = True
    itter_count = 0
    while train_flag:
        itter_count += 1
        #model = AdaBoostRegressor(n_estimators=20, random_state=42)
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        #model = LinearRegression()

        model.fit(np.array(train_set['X'].tolist()), train_set['y'])

        remaining_set = df[~df.index.isin(train_set.index)].copy()

        #print(f"Remaining set: {remaining_set.shape}")

        remaining_set['pred'] = model.predict(np.array(remaining_set['X'].tolist()))

        top_10_pred = remaining_set.nlargest(10, 'pred')
        top_10_rank_values = top_10_pred['rank'].values

        #print(top_10_pred)
        #print(top_10_rank_values)
        
        if any(int(top_10_rank_value) < 10 for top_10_rank_value in top_10_rank_values):
            train_flag = False
            print('Pizdec pabeidzas')
            print(top_10_rank_values)
        else:
            
            train_set = pd.concat([train_set, top_10_pred])
        
        #print(f"Iteration: {itter_count}")

        if itter_count >100:
            break

    return itter_count

def run_random_baseline(df, rand_state=42):
    nr_of_samples = df.shape[0]
    rng = np.random.RandomState(rand_state)
    iteration_count = 0
    while True:
        iteration_count += 1
        random_indices = rng.choice(df.index, 10, replace=False)
        top_10_random = df.loc[random_indices]
        top_10_rank_values = top_10_random['rank'].values
        if any(int(top_10_rank_value) < 10 for top_10_rank_value in top_10_rank_values):
            break
        if iteration_count > 100:
            break
    return iteration_count

def run_random_n_times(df, n=10):
    iteration_counts = []
    for i in range(n):
        iteration_count = run_random_baseline(df, rand_state=i)
        iteration_counts.append(iteration_count)
    return iteration_counts

def run_n_times(df, n=10):
    iteration_counts = []
    for i in range(n):
        iteration_count = run_bo(df, rand_state=i)
        iteration_counts.append(iteration_count)
    return iteration_counts

df_embeddings = labeled_data[['rank', 'embeddings', 'Zero-Shot Score']]
df_embeddings.columns = ['rank', 'X', 'y']

df_z_8d = labeled_data[['rank', 'latent_space_8d', 'Zero-Shot Score']]
df_z_8d.columns = ['rank', 'X', 'y']

df_z_2d = labeled_data[['rank', 'latent_space_2d', 'Zero-Shot Score']]
df_z_2d.columns = ['rank', 'X', 'y']

df_z_32d = labeled_data[['rank', 'latent_space_32d', 'Zero-Shot Score']]
df_z_32d.columns = ['rank', 'X', 'y']

n = 50
print('Running 2D')
iteration_counts_2d = run_n_times(df_z_2d, n)
print('Running 8D')
iteration_counts_8d = run_n_times(df_z_8d, n)
print('Running embeddings')
iteration_counts_embeddings = run_n_times(df_embeddings, n)
print('Running random baseline')
random_baseline_counts = run_random_n_times(df_z_2d, n)

print('Running 32D')
iteration_counts_32d = run_n_times(df_z_32d, n)

print(f'Embeddings: {iteration_counts_embeddings}')
print(f'Latent Space 8D: {iteration_counts_8d}')
print(f'Latent Space 2D: {iteration_counts_2d}')
print(f'Random Baseline: {random_baseline_counts}')
print(f'Latent Space 32D: {iteration_counts_32d}')


import matplotlib.pyplot as plt
data = {
    'Embeddings': iteration_counts_embeddings,
    'Latent Space 8D': iteration_counts_8d,
    'Latent Space 2D': iteration_counts_2d,
    'Random Baseline': random_baseline_counts
}

fig, ax = plt.subplots()
ax.boxplot(data.values(), patch_artist=True)
ax.set_xticklabels(data.keys())

for i, key in enumerate(data.keys()):
    y = data[key]
    x = np.random.normal(i + 1, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.6)

plt.ylabel('Iteration Counts')
plt.title('Box plots with scatter of iteration counts')
plt.show()














