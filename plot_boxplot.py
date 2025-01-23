# iteration_counts_embeddings = [49, 1, 4, 5, 4, 2, 14, 7, 35, 2]
# iteration_counts_8d = [46, 17, 2, 13, 93, 16, 4, 33, 25, 5]
# iteration_counts_2d = [21, 97, 50, 71, 30, 43, 68, 52, 101, 57]
# random_baseline_counts = [101, 101, 101, 101, 94, 33, 101, 67, 68, 101]

# iteration_counts_32d = [33, 7, 32, 8, 14, 22, 20, 9, 11, 5]


# # ada boost

# iteration_counts_embeddings = [6, 13, 8, 1, 8, 2, 13, 1, 5, 3]
# iteration_counts_8d = [1, 15, 31, 33, 10, 49, 17, 23, 31, 37]
# iteration_counts_2d = [1, 1, 1, 20, 17, 1, 5, 1, 43, 6]
# random_baseline_counts = [101, 101, 101, 101, 94, 33, 101, 67, 68, 101]
# iteration_counts_32d = [18, 16, 3, 1, 14, 16, 23, 13, 7, 11]

# ada boost 50
# iteration_counts_embeddings = [6, 13, 8, 1, 8, 2, 13, 1, 5, 3, 1, 3, 1, 1, 2, 1, 15, 1, 5, 3, 3, 15, 22, 6, 2, 2, 1, 5, 11, 3, 1, 13, 6, 6, 2, 8, 2, 1, 20, 4, 1, 3, 12, 10, 4, 11, 1, 1, 1, 6]
# iteration_counts_8d = [1, 15, 31, 33, 10, 49, 17, 23, 31, 37, 1, 17, 1, 20, 1, 17, 29, 53, 41, 11, 6, 23, 30, 20, 34, 34, 34, 1, 27, 27, 2, 34, 61, 63, 14, 9, 1, 32, 37, 1, 14, 27, 26, 41, 7, 35, 1, 1, 1, 19]
# iteration_counts_2d = [1, 1, 1, 20, 17, 1, 5, 1, 43, 6, 1, 23, 1, 21, 1, 19, 1, 1, 15, 1, 31, 9, 24, 55, 3, 1, 12, 44, 1, 2, 1, 6, 6, 1, 18, 1, 1, 41, 11, 18, 40, 21, 53, 1, 1, 3, 29, 9, 1, 4]
# random_baseline_counts = [101, 101, 101, 101, 94, 33, 101, 67, 68, 101, 38, 85, 46, 42, 101, 78, 45, 101, 23, 101, 21, 101, 16, 34, 101, 61, 98, 101, 66, 1, 37, 31, 101, 101, 5, 88, 53, 64, 27, 101, 28, 43, 101, 101, 76, 37, 101, 101, 101, 12]
# iteration_counts_32d = [18, 16, 3, 1, 14, 16, 23, 13, 7, 11, 21, 6, 5, 6, 15, 12, 6, 20, 3, 8, 28, 15, 1, 2, 21, 13, 1, 13, 10, 1, 2, 17, 33, 4, 36, 1, 6, 1, 16, 14, 5, 18, 34, 14, 20, 3, 1, 6, 6, 1]

# rf 50, 20 classifiers

iteration_counts_embeddings =[39, 6, 19, 3, 7, 4, 1, 2, 5, 2, 3, 14, 19, 40, 5, 7, 11, 4, 9, 12, 11, 26, 12, 29, 5, 5, 5, 5, 15, 3, 2, 7, 2, 9, 11, 10, 20, 4, 3, 2, 16, 4, 9, 18, 4, 7, 1, 14, 1, 12]
iteration_counts_8d = [10, 10, 36, 14, 37, 87, 24, 6, 64, 16, 66, 15, 15, 4, 44, 11, 31, 61, 7, 19, 62, 3, 29, 87, 17, 60, 20, 95, 42, 62, 16, 28, 5, 12, 17, 22, 46, 26, 66, 11, 61, 31, 44, 8, 51, 29, 43, 59, 40, 43]
iteration_counts_2d = [54, 68, 24, 49, 9, 101, 22, 41, 1, 77, 34, 31, 26, 87, 28, 24, 48, 1, 101, 69, 5, 101, 4, 57, 35, 41, 2, 101, 51, 27, 36, 77, 9, 64, 12, 1, 3, 1, 18, 16, 3, 4, 52, 19, 45, 20, 59, 101, 24, 5]
random_baseline_counts = [101, 101, 101, 101, 94, 33, 101, 67, 68, 101, 38, 85, 46, 42, 101, 78, 45, 101, 23, 101, 21, 101, 16, 34, 101, 61, 98, 101, 66, 1, 37, 31, 101, 101, 5, 88, 53, 64, 27, 101, 28, 43, 101, 101, 76, 37, 101, 101, 101, 12]
iteration_counts_32d = [34, 10, 20, 33, 17, 32, 10, 4, 49, 28, 19, 2, 12, 9, 2, 13, 13, 7, 11, 6, 9, 24, 10, 3, 11, 23, 1, 4, 33, 14, 24, 29, 65, 50, 7, 6, 7, 7, 6, 16, 19, 8, 2, 10, 15, 36, 48, 14, 6, 15]

import matplotlib.pyplot as plt
import numpy as np
data = {
    'Embeddings': iteration_counts_embeddings,
    'Latent 32D': iteration_counts_32d,
    'Latent 8D': iteration_counts_8d,
    'Latent 2D': iteration_counts_2d,
    'Random Baseline': random_baseline_counts
}

fig, ax = plt.subplots(figsize=(8, 5))  # Increase the width of the figure
boxprops = dict(facecolor='none', color='black')
ax.boxplot(data.values(), patch_artist=True, boxprops=boxprops, showfliers=False)
ax.set_xticklabels(data.keys())

for i, key in enumerate(data.keys()):
    y = data[key]
    x = np.random.normal(i + 1, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.4, color='black')

plt.ylabel('Iteration Counts')
plt.title('Box plots with scatter of iteration counts')
plt.savefig('project/figures/boxplot_50_rf.png')