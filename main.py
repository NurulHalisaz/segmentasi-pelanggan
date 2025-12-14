import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Customer Profiling - Final", layout="wide")
st.title("📊 Customer Profiling & Cluster Assignment")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_customers_cleaned_raw_for_clustering.csv")

df = load_data()

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

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# TRAIN K-MEANS (FIXED K)
# =========================
kmeans = KMeans(n_clusters=4, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# =========================
# PCA FOR VISUALIZATION
# =========================
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df["PC1"] = pca_result[:, 0]
df["PC2"] = pca_result[:, 1]

# =========================
# VISUALISASI CLUSTER
# =========================
st.subheader("Visualisasi Cluster Pelanggan (PCA)")

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(
    df["PC1"],
    df["PC2"],
    c=df["cluster"],
    s=15
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Customer Profiling Clustering (K = 4)")
st.pyplot(fig)

# =========================
# PROFIL CLUSTER
# =========================
st.subheader("Profil Rata-Rata Tiap Cluster")
cluster_profile = df.groupby("cluster")[features].mean().round(2)
st.dataframe(cluster_profile)

# =========================
# INPUT DATA BARU
# =========================
st.subheader("🔎 Cek Cluster untuk Customer Baru")

with st.form("input_customer"):
    total_spent = st.number_input("Total Spent", min_value=0.0)
    avg_order_value = st.number_input("Average Order Value", min_value=0.0)
    recency = st.number_input("Recency (hari)", min_value=0)
    tenure = st.number_input("Tenure (hari)", min_value=0)
    loyalty_points = st.number_input("Loyalty Points", min_value=0)
    support_tickets = st.number_input("Support Tickets", min_value=0)
    churn_risk = st.slider("Churn Risk", 0.0, 1.0, 0.2)

    submit = st.form_submit_button("Tentukan Cluster")

if submit:
    new_data = pd.DataFrame([[
        total_spent, avg_order_value, recency, tenure,
        loyalty_points, support_tickets, churn_risk
    ]], columns=features)

    new_scaled = scaler.transform(new_data)
    cluster_result = kmeans.predict(new_scaled)[0]

    st.success(f"Customer ini masuk ke **Cluster {cluster_result}**")

    st.write("Karakteristik rata-rata cluster tersebut:")
    st.dataframe(cluster_profile.loc[[cluster_result]])

# =========================
# INTERPRETASI
# =========================
st.subheader("📌 Interpretasi Singkat")
st.write("""
- Model menggunakan K-Means dengan K = 4 berdasarkan Elbow Method.
- Setiap cluster merepresentasikan segmen pelanggan dengan karakteristik berbeda.
- Data customer baru dipetakan ke cluster terdekat tanpa melatih ulang model.
""")

