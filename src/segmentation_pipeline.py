"""
Customer Segmentation Pipeline
--------------------------------

"""

import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


# DATA LOADING


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw customer dataset
    """
    return pd.read_csv(path)



# PREPROCESSING & SCALING


def preprocess_and_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess data exactly as notebooks
    - Select numeric features
    - Handle missing values
    - Apply StandardScaler
    """


    df = df[
        ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    ].copy()

    # Ensure numeric type
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    # Safety check
    if df.shape[0] == 0:
        raise ValueError("No valid data available after preprocessing.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

    return df_scaled



# KMEANS CLUSTERING


def apply_kmeans(data: pd.DataFrame, n_clusters: int):
    """
    Apply KMeans clustering
    """
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(data)
    return labels, model


def evaluate_kmeans(data: pd.DataFrame, labels):
    """
    Compute Silhouette Score
    """
    return silhouette_score(data, labels)



# HIERARCHICAL CLUSTERING


def apply_hierarchical(data: pd.DataFrame, n_clusters: int):
    """
    Apply Agglomerative Clustering
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )
    labels = model.fit_predict(data)
    return labels



# DBSCAN CLUSTERING


def apply_dbscan(data: pd.DataFrame, eps: float, min_samples: int):
    """
    Apply DBSCAN clustering
    """
    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )
    labels = model.fit_predict(data)
    return labels



# SAVE & LOAD UTILITIES (OPTIONAL)


def save_object(obj, path: str):
    """
    Save Python object using pickle
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_object(path: str):
    """
    Load Python object from pickle file
    """
    with open(path, "rb") as f:
        return pickle.load(f)
