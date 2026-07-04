# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from mpl_toolkits.mplot3d import Axes3D

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

# Keep only numeric columns
df = df.select_dtypes(
    include=['number']
)

# Remove missing values
df = df.dropna()

# Standardize data
scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

# -----------------------------
# PCA for 2D Visualization
# -----------------------------

pca_2d = PCA(
    n_components=2
)

principal_components_2d = pca_2d.fit_transform(
    scaled_data
)

pca_df_2d = pd.DataFrame(
    principal_components_2d,
    columns=["PC1", "PC2"]
)

print("Explained Variance Ratio (2D):")
print(
    pca_2d.explained_variance_ratio_
)

# 2D Plot
plt.figure(figsize=(8,6))

plt.scatter(
    pca_df_2d["PC1"],
    pca_df_2d["PC2"]
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title(
    "PCA 2D Visualization"
)

plt.show()

# -----------------------------
# PCA for 3D Visualization
# -----------------------------

pca_3d = PCA(
    n_components=3
)

principal_components_3d = pca_3d.fit_transform(
    scaled_data
)

pca_df_3d = pd.DataFrame(
    principal_components_3d,
    columns=["PC1","PC2","PC3"]
)

print("\nExplained Variance Ratio (3D):")
print(
    pca_3d.explained_variance_ratio_
)

# 3D Plot
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(
    111,
    projection='3d'
)

ax.scatter(
    pca_df_3d["PC1"],
    pca_df_3d["PC2"],
    pca_df_3d["PC3"]
)

ax.set_xlabel(
    "Principal Component 1"
)

ax.set_ylabel(
    "Principal Component 2"
)

ax.set_zlabel(
    "Principal Component 3"
)

plt.title(
    "PCA 3D Visualization"
)

plt.show()