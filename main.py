import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Customer Profiling", layout="wide")

# =========================
# TITLE & INTRO
# =========================
st.title("📊 Customer Profiling")

st.markdown("""
**Customer Profiling** adalah proses analisis data pelanggan untuk memahami
pola perilaku, karakteristik, dan nilai pelanggan.

Aplikasi ini menampilkan **dua analisis terpisah**:
1. **Segmentasi Pelanggan (Clustering)** → Mengelompokkan pelanggan berdasarkan kemiripan karakteristik
2. **Prediksi Spending Score (Regresi)** → Memprediksi tingkat pengeluaran pelanggan
""")

# =========================
# DATASET UPLOAD
# =========================
st.header("📁 Upload Dataset")
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset berhasil diupload")

    st.subheader("🔍 Preview Data")
    st.dataframe(df.head())

    # =========================
    # FEATURE SELECTION
    # =========================
    features = [
        'total_spent',
        'avg_order_value',
        'Recency',
        'Tenure',
        'loyalty_points',
        'support_tickets',
        'churn_risk'
    ]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Kolom berikut tidak ditemukan dalam dataset: {missing_features}")
        st.stop()

    X = df[features]

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ==========================================================
    # ANALISIS 1: CLUSTERING
    # ==========================================================
    st.header("🔹 Analisis 1: Segmentasi Pelanggan (Clustering)")

    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df['Cluster'] = cluster_labels

    sil_score = silhouette_score(X_scaled, cluster_labels)

    st.write(f"**Jumlah Cluster:** {k}")
    st.write(f"**Silhouette Score:** {sil_score:.4f}")

    st.subheader("Distribusi Cluster")
    st.bar_chart(df['Cluster'].value_counts())

    st.subheader("Contoh Data dengan Label Cluster")
    st.dataframe(df.head())

    # ==========================================================
    # ANALISIS 2: REGRESI
    # ==========================================================
    st.header("🔹 Analisis 2: Prediksi Spending Score (Regresi)")

    st.markdown("""
    Pada analisis ini digunakan metode **regresi** untuk memprediksi
    **Spending Score**, yang direpresentasikan oleh variabel **total_spent**.
    Model yang digunakan adalah **Random Forest Regressor** karena mampu
    menangkap hubungan non-linear pada data pelanggan.
    """)

    # Target & fitur regresi
    y = df['total_spent']
    X_reg = df[features[1:]]  # tanpa total_spent

    X_reg_scaled = scaler.fit_transform(X_reg)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_reg_scaled, y, test_size=0.2, random_state=42
    )

    # Model regresi
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"📉 **MSE:** {mse:,.2f}")
    st.write(f"📉 **RMSE:** {rmse:,.2f}")
    st.write(f"📈 **R² Score:** {r2:.4f}")

    # =========================
    # VISUALISASI REGRESI
    # =========================
    st.subheader("📊 Visualisasi Prediksi")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual Spending Score")
    ax.set_ylabel("Predicted Spending Score")
    ax.set_title("Actual vs Predicted Spending Score")
    st.pyplot(fig)

    # =========================
    # INTERPRETASI SINGKAT
    # =========================
    st.info("""
    🔍 **Interpretasi:**
    - Nilai **R²** menunjukkan seberapa baik model menjelaskan variasi pengeluaran pelanggan.
    - Semakin mendekati **1**, semakin baik performa model.
    - Random Forest dipilih karena lebih robust terhadap data non-linear dan outlier.
    """)

else:
    st.info("Silakan upload dataset CSV untuk memulai analisis.")
