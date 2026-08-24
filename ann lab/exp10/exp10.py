import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load Dataset
df = pd.read_csv(r"C:\Users\M R Computer\OneDrive\Desktop\ann lab\exp10\bank_customers.csv")

print("Dataset Preview:")
print(df.head())

# Select Features
X = df[['Age',
        'Balance',
        'Transactions',
        'Loans',
        'CreditScore',
        'Savings']]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.show()

# Apply KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# Add Cluster Column
df['Cluster'] = clusters

print("\nClustered Dataset:")
print(df)

# Print centroids in scaled feature space
centroids_scaled = kmeans.cluster_centers_
print("\nCluster Centroids (scaled):")
print(centroids_scaled)

# Print centroids in original feature space
centroids_unscaled = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_unscaled, columns=X.columns)
centroids_df.index.name = 'Cluster'
print("\nCluster Centroids (original scale):")
print(centroids_df)

# PCA for Visualization
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(centroids_scaled)

# Visualize Clusters
plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    cmap='viridis',
    alpha=0.7,
    edgecolor='k'
)
plt.scatter(
    centroids_pca[:,0],
    centroids_pca[:,1],
    c='red',
    marker='X',
    s=200,
    label='Centroids'
)

plt.title("Bank Customer Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)

plt.show()

# Cluster Summary
print("\nCluster Summary")

summary = df.groupby('Cluster').mean()

print(summary)

# Number of Customers per Cluster
print("\nCustomers in Each Cluster")

print(df['Cluster'].value_counts())

# Cluster Centroids
centroids_scaled = kmeans.cluster_centers_
print("\nCluster Centroids (scaled):")
print(centroids_scaled)

centroids_unscaled = scaler.inverse_transform(centroids_scaled)
print("\nCluster Centroids (original scale):")
print(pd.DataFrame(centroids_unscaled, columns=X.columns))