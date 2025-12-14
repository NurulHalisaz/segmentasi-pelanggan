import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score

# PAGE CONFIG
st.set_page_config(page_title="Customer Profiling", layout="wide")


# TITLE & INTRO
st.title("📊 Customer Profiling")

st.markdown("""
**Customer Profiling** adalah proses menganalisis data pelanggan untuk memahami
pola perilaku, karakteristik, dan nilai pelanggan.  
Pada aplikasi ini dilakukan dua analisis terpisah:

1. **Segmentasi Pelanggan ** → Mengelompokkan pelanggan berdasarkan kemiripan karakteristik  
2. **Prediksi Spending Score** → Memprediksi tingkat pengeluaran pelanggan
""")

# DATASET UPLOAD
st.header("📁 Upload Dataset")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diupload")

    st.subheader("Preview Data")
    st.dataframe(df.head())

    # FEATURE SELECTION
    features = [
        'total_spent',
        'avg_order_value',
        'Recency',
        'Tenure',
        'loyalty_points',
        'support_tickets',
        'churn_risk'
    ]

    # Pastikan fitur tersedia
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Kolom berikut tidak ditemukan dalam dataset: {missing_features}")
        st.stop()

    X = df[features]

    # SCALING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ANALISIS 1: 
    st.header("🔹 Analisis 1: Segmentasi Pelanggan")

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df['Cluster'] = cluster_labels

    sil_score = silhouette_score(X_scaled, cluster_labels)

    st.write(f"**Jumlah Cluster:** {k}")
    st.write(f"**Silhouette Score:** {sil_score:.4f}")

    st.subheader("Distribusi Cluster")
    st.bar_chart(df['Cluster'].value_counts())

    st.subheader("Contoh Data per Cluster")
    st.dataframe(df.head())

    # ANALISIS 2: 
    st.header("🔹 Analisis 2: Prediksi Spending Score ")

    st.markdown("""
    Pada analisis ini digunakan metode **regresi** untuk memprediksi
    **Spending Score**, yang direpresentasikan oleh variabel **total_spent**.
    """)

    # Target & fitur regresi
    y = df['total_spent']
    X_reg = df[features[1:]]  # tanpa total_spent

    X_reg_scaled = scaler.fit_transform(X_reg)

    model = LinearRegression()
    model.fit(X_reg_scaled, y)

    y_pred = model.predict(X_reg_scaled)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # VISUALISASI REGRESI

    fig, ax = plt.subplots()
    ax.scatter(y, y_pred)
    ax.set_xlabel("Actual Spending Score")
    ax.set_ylabel("Predicted Spending Score")
    ax.set_title("Actual vs Predicted Spending Score")
    st.pyplot(fig)

else:
    st.info("Silakan upload dataset CSV untuk memulai analisis.")
