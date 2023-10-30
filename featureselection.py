from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
# Create a PCA instance
pca = PCA(n_components=2)  # You can specify the number of components you want to extract
data = pd.read_csv('resampled.csv')

# Fit and transform the data to the first two principal components
principal_components = pca.fit_transform(data)
# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)
import matplotlib.pyplot as plt

# Plot cumulative explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(np.cumsum(explained_variance_ratio))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Set a threshold for cumulative explained variance
desired_variance = 0.95  # Example threshold of 95%
n_components = np.argmax(np.cumsum(explained_variance_ratio) >= desired_variance) + 1

print("Number of components selected:", n_components)
