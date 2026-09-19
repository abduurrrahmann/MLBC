import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Load the Iris dataset

iris = load_iris()
X = iris.data
y = iris.target

#Standardize the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply PCA and reduce 4 features to 2 components

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#Display explained variance

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", sum(pca.explained_variance_ratio_))

# Plot the principal components

plt.figure(figsize=(8,6))
for i, name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=name)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Iris Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()