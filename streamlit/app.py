import sys
import os


# Add project root to Python path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Libraries

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


# Project Imports

from src.segmentation_pipeline import (
    preprocess_and_scale,
    apply_kmeans,
    apply_hierarchical,
    apply_dbscan,
    evaluate_kmeans
)


# Page Configuration

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #03A9F4;'>
        Customer Segmentation Project
    </h1>
    """,
    unsafe_allow_html=True
)


# DATASET UPLOAD (SIDEBAR)

st.sidebar.header(" Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
else:
    # Default dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "Mall_Customers.csv")
    df = pd.read_csv(DATA_PATH)

st.subheader(" Raw Data")
st.dataframe(df.head())


# PREPROCESSING 

df_scaled = preprocess_and_scale(df)


# ALGORITHM SELECTION 

st.sidebar.header(" Clustering Options")

algo = st.sidebar.selectbox(
    "Choose Algorithm",
    ["K-Means", "Hierarchical", "DBSCAN"]
)


# CLUSTERING

if algo == "K-Means":
    k = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4)
    labels, model = apply_kmeans(df_scaled, k)
    score = evaluate_kmeans(df_scaled, labels)

elif algo == "Hierarchical":
    k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    labels = apply_hierarchical(df_scaled, k)
    score = None

else:
    eps = st.sidebar.slider("EPS", 0.1, 2.0, 1.0)
    min_samples = st.sidebar.slider("Min Samples", 2, 10, 5)
    labels = apply_dbscan(df_scaled, eps, min_samples)
    score = None


# RESULTS

df_result = df.copy()
df_result["Cluster"] = labels

st.subheader(" Clustered Data")
st.dataframe(df_result.head())

if score is not None:
    st.metric("Silhouette Score", round(score, 3))


# CLUSTER DISTRIBUTION

st.subheader(" Cluster Distribution")

col1, col2 = st.columns(2)

with col1:
    bar_fig = px.bar(
        df_result["Cluster"].value_counts().sort_index(),
        labels={"value": "Customers", "index": "Cluster"},
        title="Customers per Cluster"
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with col2:
    pie_fig = px.pie(
        df_result,
        names="Cluster",
        title="Cluster Share"
    )
    st.plotly_chart(pie_fig, use_container_width=True)


# PCA VISUALIZATION

st.subheader(" Cluster Visualization (PCA)")

pca = PCA(n_components=2)
pca_data = pca.fit_transform(df_scaled)

df_pca = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
df_pca["Cluster"] = labels

fig = px.scatter(
    df_pca,
    x="PC1",
    y="PC2",
    color="Cluster",
    title="PCA Cluster Visualization"
)

st.plotly_chart(fig, use_container_width=True)


# CLUSTER PROFILING

st.subheader(" Cluster Profiling")

profile = df_result.groupby("Cluster")[
    ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
].mean()

st.dataframe(profile)


# DOWNLOAD RESULTS

st.subheader(" Download Results")

csv = df_result.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Segmented Customers",
    csv,
    "customer_segments.csv",
    "text/csv"
)


# CUSTOMER SEARCH

st.subheader("🔍 Search Customer")

if "CustomerID" in df_result.columns:
    cust_id = st.number_input("Enter Customer ID", min_value=1)

    if cust_id in df_result["CustomerID"].values:
        st.success("Customer Found ")
        st.dataframe(df_result[df_result["CustomerID"] == cust_id])
    else:
        st.warning("Customer ID not found")
