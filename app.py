import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Aplikasi Analisis & ML Bansos Tanjungpinang",
    page_icon="📊",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
    <style>
        .main-title { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; }
        .sub-title { font-size: 1.1rem; color: #475569; margin-bottom: 20px; }
        .card { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border: 1px solid #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

# 1. Load Data
@st.cache_data
def load_data():
    df_clean = pd.read_excel('data_bansos_bersih.xlsx')
    df_pkh = pd.read_excel('pkh.xlsx')
    df_bpnt = pd.read_excel('bpnt.xlsx')
    return df_clean, df_pkh, df_bpnt

try:
    df, df_pkh, df_bpnt = load_data()
except Exception as e:
    st.error(f"Gagal memuat file dataset (.xlsx): {e}")
    st.stop()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Menu Navigasi Aplikasi")
st.sidebar.markdown("---")
menu = st.sidebar.radio(
    "Pilih Halaman:",
    [
        "1. Overview & Data Collection",
        "2. Metode 1: K-Means Clustering",
        "3. Metode 2: Linear Regression",
        "4. Tren Bulanan (PKH & BPNT)",
        "5. Kesimpulan & Rekomendasi"
    ]
)

# ==========================================
# MENU 1: OVERVIEW & DATA COLLECTION
# ==========================================
if menu == "1. Overview & Data Collection":
    st.markdown('<p class="main-title">📊 Dashboard Analisis Bansos Kota Tanjungpinang</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Implementasi Metode K-Means Clustering & Linear Regression untuk Evaluasi Sebaran Bantuan Sosial</p>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="card"><h4>Total Wilayah</h4><h2>{len(df)} Kelurahan</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><h4>Akumulasi Total Bansos</h4><h2>{df["Total Bnasos"].sum():,} Jiwa</h2></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><h4>Rata-rata per Kelurahan</h4><h2>{df["Total Bnasos"].mean():,.1f}</h2></div>', unsafe_allow_html=True)

    st.subheader("📋 Dataset Bersih Utama (`data_bansos_bersih.xlsx`)")
    st.dataframe(df[['Kecamatan', 'Kelurahan', 'Total BPNT 2025', 'Total PKH 2025', 'Total Bnasos']], use_container_width=True)

# ==========================================
# MENU 2: METODE 1 — K-MEANS CLUSTERING
# ==========================================
elif menu == "2. Metode 1: K-Means Clustering":
    st.markdown('<p class="main-title">🤖 Metode 1: K-Means Clustering</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Mengelompokkan 18 kelurahan berdasarkan pola penerima bantuan sosial[cite: 1]</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Proses Clustering
    X = df[['Total BPNT 2025', 'Total PKH 2025']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Mapping otomatis kategori berdasarkan rata-rata total bansos
    cluster_means = df.groupby('Cluster')['Total Bnasos'].mean().sort_values()
    sorted_clusters = cluster_means.index.tolist()
    
    cluster_mapping = {
        sorted_clusters[0]: 'Kelompok Penerima Rendah',
        sorted_clusters[1]: 'Kelompok Penerima Sedang',
        sorted_clusters[2]: 'Kelompok Penerima Tinggi'
    }
    df['Kategori_Kelompok'] = df['Cluster'].map(cluster_mapping)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.subheader("Hasil Pengelompokan Kelurahan")
        st.dataframe(df[['Kecamatan', 'Kelurahan', 'Total Bnasos', 'Kategori_Kelompok']], use_container_width=True, height=400)

    with col_b:
        st.subheader("Visualisasi Sebaran Klaster")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            x='Total BPNT 2025', y='Total PKH 2025', 
            hue='Kategori_Kelompok', palette={'Kelompok Penerima Rendah':'#3B82F6', 'Kelompok Penerima Sedang':'#F59E0B', 'Kelompok Penerima Tinggi':'#EF4444'}, 
            data=df, s=120, ax=ax, edgecolor='black'
        )
        ax.set_title('Klasterisasi Beban Wilayah Bansos', fontsize=11, fontweight='bold')
        ax.set_xlabel('Total BPNT 2025')
        ax.set_ylabel('Total PKH 2025')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title='Kategori')
        st.pyplot(fig)

    st.info("""
    **Analisis & Tindak Lanjut K-Means:**
    * **Kelompok Penerima Tinggi:** Mencakup wilayah seperti **Batu Sembilan** dan **Pinang Kencana**[cite: 1], yang memerlukan perhatian khusus, prioritas intervensi anggaran, serta pengawasan validasi data yang ketat.
    * **Kelompok Penerima Sedang & Rendah:** Menunjukkan wilayah dengan beban menengah hingga pusat kota lama/pesisir[cite: 1].
    """)

# ==========================================
# MENU 3: METODE 2 — LINEAR REGRESSION
# ==========================================
elif menu == "3. Metode 2: Linear Regression":
    st.markdown('<p class="main-title">📈 Metode 2: Linear Regression</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Menganalisis hubungan numerik dan memprediksi jumlah penerima PKH berdasarkan BPNT[cite: 1]</p>', unsafe_allow_html=True)
    st.markdown("---")

    X_reg = df[['Total BPNT 2025']]
    y_reg = df['Total PKH 2025']

    reg = LinearRegression()
    reg.fit(X_reg, y_reg)
    y_pred = reg.predict(X_reg)

    r2 = r2_score(y_reg, y_pred)
    mae = mean_absolute_error(y_reg, y_pred)
    rmse = np.sqrt(mean_squared_error(y_reg, y_pred))

    col_e1, col_e2, col_e3 = st.columns(3)
    col_e1.metric("R-Squared ($R^2$ Score)", f"{r2:.4f}")
    col_e2.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col_e3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

    st.markdown("---")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("Kurva Regresi Linear")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(X_reg, y_reg, color='#2563EB', label='Data Aktual', s=80)
        ax.plot(X_reg, y_pred, color='#DC2626', linewidth=2, label='Garis Prediksi')
        ax.set_title('Hubungan BPNT terhadap PKH', fontsize=11, fontweight='bold')
        ax.set_xlabel('Total BPNT 2025')
        ax.set_ylabel('Total PKH 2025')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig)

    with col_p2:
        st.subheader("Persamaan & Interpretasi Model")
        st.success(f"""
        **Persamaan Regresi:**  
        `Total PKH = {reg.coef_[0]:.4f} × (Total BPNT) + {reg.intercept_:.2f}`

        **Interpretasi Analitis:**
        * Nilai $R^2$ sebesar **{r2:.4f}** (93.4%) membuktikan adanya hubungan linier positif yang sangat kuat[cite: 1].
        * Model ini dapat diandalkan untuk memprediksi atau mengestimasi alokasi peserta PKH apabila terjadi perubahan kuota data BPNT di masa mendatang[cite: 1].
        """)

# ==========================================
# MENU 4: TREN BULANAN (PKH & BPNT)
# ==========================================
elif menu == "4. Tren Bulanan (PKH & BPNT)":
    st.markdown('<p class="main-title">📅 Tren Bulanan Penyaluran Bantuan</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Analisis perkembangan data historis bulanan (Januari – Desember 2025)</p>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["📌 Program PKH", "🛒 Program BPNT"])
    months = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

    with tab1:
        st.dataframe(df_pkh, use_container_width=True)
        pkh_sum = df_pkh[months].sum()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(months, pkh_sum, marker='o', color='#2563EB', linewidth=2)
        ax.set_title('Tren Total Penerima PKH per Bulan', fontsize=11, fontweight='bold')
        plt.xticks(rotation=45)
        ax.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig)

    with tab2:
        st.dataframe(df_bpnt, use_container_width=True)
        bpnt_sum = df_bpnt[months].sum()
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.plot(months, bpnt_sum, marker='s', color='#DC2626', linewidth=2)
        ax2.set_title('Tren Total Penerima BPNT per Bulan', fontsize=11, fontweight='bold')
        plt.xticks(rotation=45)
        ax2.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig2)

# ==========================================
# MENU 5: KESIMPULAN & REKOMENDASI
# ==========================================
elif menu == "5. Kesimpulan & Rekomendasi":
    st.markdown('<p class="main-title">💡 Kesimpulan & Rekomendasi Kebijakan</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Implikasi hasil analisis data sains untuk perbaikan program bantuan sosial daerah</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    ### 1. Kesimpulan Utama
    * **Pemerataan Wilayah:** Berdasarkan hasil klasterisasi K-Means, terbukti bahwa beban bantuan sosial **belum merata sepenuhnya**, melainkan terkonsentrasi di *Kelompok Penerima Tinggi* (Kecamatan Tanjungpinang Timur, khususnya Kelurahan Batu Sembilan & Pinang Kencana)[cite: 1].
    * **Hubungan & Prediksi:** Model *Linear Regression* berhasil memvalidasi korelasi kuat antarprogram ($R^2 = 0.934$), menunjukkan konsistensi sasaran rumah tangga penerima manfaat[cite: 1].

    ### 2. Rekomendasi bagi Pemerintah Daerah
    * Menjadikan wilayah dalam **Kelompok Penerima Tinggi** sebagai prioritas utama dalam alokasi intervensi pemberdayaan dan verifikasi berkala (DTSEN)[cite: 1].
    """)
    st.success("Aplikasi siap digunakan untuk presentasi dan pelaporan Project 2 Anda!")