import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
import pickle
warnings.filterwarnings('ignore')


def load_data(path):
    """Load the Super Store dataset from the given path."""
    return pd.read_excel(path)


def preprocess_data(df,features):
    """select features,handle missing values, and scale the data."""
    x=df[features].copy()
    x=x.dropna()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    return x_scaled,x.index
# ...existing code...

from sklearn.metrics import davies_bouldin_score  # Add this import

def plot_elbow(x_scaled):
    inertia = []
    for k in range(2, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(x_scaled)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(8, 4))  # Correct argument name
    plt.plot(range(2, 11), inertia, marker='o')
    plt.xlabel('Number of clusters')  # Typo fixed
    plt.ylabel('Inertia')
    plt.title('Elbow Method for optimal k')
    plt.savefig('elbow_plot.png')  
    plt.show()

def kmeans(X_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Fix variable name
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans

def evaluate_clustering(X_scaled, labels):
    print('/n--- Clustering Validation Metrics ---')  # Fix newline
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
    print(f'Silhouette Score: {sil:.3f}')
    print(f'Calinski-Harabasz Index: {ch:.3f}')
    print(f'Davies-Bouldin Index: {db:.3f}')

def cluster_profiling(df, features, cluster_col):
    for c in sorted(df[cluster_col].unique()):
        print(f'/nCluster {c}:')  # Fix newline
        print(df[df[cluster_col] == c][features].describe())

def main():
    data_path = 'c:/Users/DELL/Downloads/Sample - Superstore.xls'  # Update path if needed
    features = ['Sales', 'Quantity', 'Discount', 'Profit']
    df = load_data(data_path)
    X_scaled, valid_idx = preprocess_data(df, features)
    print('Data loaded and preprocessed.')

    print('/n--- Elbow Method ---')  # Fix newline
    plot_elbow(X_scaled)

    n_clusters = 4  # Set based on elbow plot
    labels, kmeans_model = kmeans(X_scaled, n_clusters)  # Avoid overwriting function
    df.loc[valid_idx, 'Cluster'] = labels

    evaluate_clustering(X_scaled, labels)

    print('/n--- Cluster Means ---')  # Fix newline
    print(df.groupby('Cluster')[features].mean())

    print('/n--- Cluster Visualization ---')  # Fix newline
    visualize_clusters(df.loc[valid_idx], features, 'Cluster')

    print('/n--- Cluster Profiling ---')  # Fix newline
    cluster_profiling(df.loc[valid_idx], features, 'Cluster')

if __name__ == '__main__':  # Fix main guard
   main()