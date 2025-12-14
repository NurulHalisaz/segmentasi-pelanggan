import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# CONFIG
st.set_page_config(page_title="Customer Profiling Clustering", layout="wide")

st.title("📊 Customer Profiling dengan K-Means Clustering")
st.write("Aplikasi ini digunakan untuk segmentasi pelanggan berbasis perilaku dan nilai bisnis.")

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_customers_cleaned_raw_for_clustering.csv")
    return df

df = load_data()

st.subheader("Preview Dataset")
st.dataframe(df.head())

# FEATURE SELECTION
features = [
    "total_spent",
    "avg_order_value",
    "Recency",
    "Tenure",
    "loyalty_points",
    "support_tickets",
    "churn_risk"
]

X = df[features]

# SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PILIH JUMLAH CLUSTER
st.sidebar.header("Pengaturan Model")
k = st.sidebar.slider("Jumlah Cluster (K)", 2, 6, 4)

# K-MEANS
kmeans = KMeans(n_clusters=k, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

# VISUALISASI
st.subheader("Visualisasi Cluster (PCA)")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    df["PC1"],
    df["PC2"],
    c=df["cluster"],
    s=15
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Visualisasi Customer Profiling")

st.pyplot(fig)

# PROFIL CLUSTER
st.subheader("Profil Rata-rata Tiap Cluster")

cluster_profile = df.groupby("cluster")[features].mean().round(2)
st.dataframe(cluster_profile)

# INTERPRETASI SINGKAT

st.subheader("Interpretasi Singkat")
st.write("""
- Setiap warna merepresentasikan segmen pelanggan yang berbeda.
- Cluster dibentuk berdasarkan kemiripan perilaku dan nilai pelanggan.
- Profil cluster membantu menentukan strategi bisnis yang lebih tepat sasaran.
""")
