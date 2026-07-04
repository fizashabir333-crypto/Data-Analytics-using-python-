# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_excel(
    "data.xlsx",
    engine="openpyxl"
)

# -----------------------------
# Data preprocessing
# -----------------------------

# Keep only numerical columns
df = df.select_dtypes(
    include=['number']
)

# Remove missing values
df = df.dropna()

# Standardize data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

# -----------------------------
# K-Means Clustering
# -----------------------------

kmeans = KMeans(
    n_clusters=3,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(
    scaled_data
)

# Silhouette Score
kmeans_score = silhouette_score(
    scaled_data,
    kmeans_labels
)

print("K-Means Silhouette Score:")
print(kmeans_score)

# -----------------------------
# Scatter Plot of Clusters
# -----------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    scaled_data[:,0],
    scaled_data[:,1],
    c=kmeans_labels
)

plt.title(
    "K-Means Clustering"
)

plt.xlabel(
    "Feature 1"
)

plt.ylabel(
    "Feature 2"
)

plt.show()

# -----------------------------
# Hierarchical Clustering
# -----------------------------

hierarchical = AgglomerativeClustering(
    n_clusters=3
)

hierarchical_labels = hierarchical.fit_predict(
    scaled_data
)

# Silhouette Score
hierarchical_score = silhouette_score(
    scaled_data,
    hierarchical_labels
)

print(
    "\nHierarchical Silhouette Score:"
)

print(
    hierarchical_score
)

# -----------------------------
# Dendrogram
# -----------------------------

plt.figure(figsize=(10,6))

linkage_matrix = linkage(
    scaled_data,
    method='ward'
)

dendrogram(
    linkage_matrix
)

plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel(
    "Data Points"
)

plt.ylabel(
    "Distance"
)

plt.show()