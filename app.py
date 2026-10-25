from pathlib import Path
import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# =========================================================
# KONFIGURASI APLIKASI
# =========================================================

st.set_page_config(
    page_title="Dashboard Bansos Tanjungpinang 2025",
    page_icon="📊",
    layout="wide"
)

MONTHS = [
    "Januari", "Februari", "Maret", "April",
    "Mei", "Juni", "Juli", "Agustus",
    "September", "Oktober", "November", "Desember"
]

MONTH_LOOKUP = {m.lower(): m for m in MONTHS}


# =========================================================
# DESAIN DASHBOARD
# =========================================================

st.markdown("""
<style>
    .stApp {
        background: #f4f7fb;
    }

    [data-testid="stHeader"] {
        background: rgba(244,247,251,0.9);
    }

    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 18px;
        background: linear-gradient(120deg,#102a43,#176b87);
        color: white;
        margin-bottom: 1rem;
    }

    .hero h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
    }

    .hero p {
        color: #d9edf5;
        margin: .35rem 0 0 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e4ebf2;
        padding: 14px 16px;
        border-radius: 14px;
    }

    .small-note {
        color: #536779;
        font-size: .9rem;
    }
</style>

<div class="hero">
    <h1>Dashboard Bantuan Sosial Kota Tanjungpinang</h1>
    <p>
        Analisis bulanan PKH dan BPNT/Sembako |
        Tahun 2025 | Tingkat kecamatan dan kelurahan
    </p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# FUNGSI PENGOLAHAN DATA
# =========================================================

def normalize_columns(df):
    """Menyeragamkan nama kolom bulan."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}

    for col in df.columns:
        key = re.sub(r"\s+", " ", str(col).strip()).lower()

        if key in {m.lower() for m in MONTHS}:
            rename[col] = MONTH_LOOKUP[key]

    return df.rename(columns=rename)


def parse_workbook(file_obj, program_name):
    """
    Membaca file Excel dan mengambil baris kelurahan.

    Baris kecamatan dan Total tidak dijumlahkan ulang
    agar tidak terjadi penghitungan ganda.
    """

    raw = pd.read_excel(file_obj, sheet_name=0)
    raw = normalize_columns(raw)

    if not len(raw.columns):
        raise ValueError(
            f"File {program_name} tidak memiliki kolom."
        )

    label_col = raw.columns[0]

    missing_months = [
        m for m in MONTHS if m not in raw.columns
    ]

    if missing_months:
        raise ValueError(
            f"Kolom bulan tidak lengkap pada {program_name}: "
            + ", ".join(missing_months)
        )

    records = []
    reported_total = {}
    current_district = None

    for _, row in raw.iterrows():
        label_value = row.get(label_col)

        if pd.isna(label_value):
            continue

        label = re.sub(
            r"\s+", " ", str(label_value)
        ).strip()

        lower = label.lower()

        vals = {
            m: pd.to_numeric(row[m], errors="coerce")
            for m in MONTHS
        }

        # Baris subtotal kecamatan
        if lower.startswith("kec.") or lower.startswith(
            "kecamatan "
        ):
            current_district = re.sub(
                r"^(kec\.|kecamatan)\s*",
                "",
                label,
                flags=re.I
            ).strip()

            continue

        # Baris total keseluruhan
        if lower == "total":
            reported_total = vals
            continue

        # Lewati baris yang belum dapat dikaitkan
        # dengan kecamatan
        if current_district is None:
            continue

        rec = {
            "Kecamatan": current_district,
            "Kelurahan": label,
            "Program": program_name
        }

        rec.update(vals)
        records.append(rec)

    detail = pd.DataFrame(records)

    if detail.empty:
        raise ValueError(
            f"Tidak ditemukan baris kelurahan "
            f"pada file {program_name}."
        )

    for m in MONTHS:
        detail[m] = pd.to_numeric(
            detail[m], errors="coerce"
        )

    detail[MONTHS] = detail[MONTHS].fillna(0)

    # Statistik perubahan Januari sampai Desember
    detail["Rata-rata bulanan"] = (
        detail[MONTHS].mean(axis=1)
    )

    detail["Januari-Desember"] = (
        detail["Desember"] - detail["Januari"]
    )

    detail["Perubahan %"] = np.where(
        detail["Januari"] != 0,
        detail["Januari-Desember"]
        / detail["Januari"] * 100,
        np.nan
    )

    # Total yang dihitung dari baris kelurahan
    calc_total = detail[MONTHS].sum(axis=0)

    total_check = pd.DataFrame({
        "Bulan": MONTHS,
        "Total hasil penjumlahan kelurahan": [
            float(calc_total[m]) for m in MONTHS
        ],
        "Total tercantum di file": [
            float(reported_total.get(m, np.nan))
            if pd.notna(reported_total.get(m, np.nan))
            else np.nan
            for m in MONTHS
        ]
    })

    total_check["Selisih (kelurahan - file)"] = (
        total_check["Total hasil penjumlahan kelurahan"]
        - total_check["Total tercantum di file"]
    )

    total_check["Program"] = program_name

    return detail, total_check, raw


def format_int(value):
    """Format angka dengan pemisah ribuan Indonesia."""
    if pd.isna(value):
        return "—"

    return f"{value:,.0f}".replace(",", ".")


def format_pct(value):
    """Format persentase."""
    if pd.isna(value):
        return "—"

    return f"{value:+.1f}%".replace(".", ",")


def to_long(df):
    """Mengubah data bulanan menjadi format panjang."""
    return df.melt(
        id_vars=["Kecamatan", "Kelurahan", "Program"],
        value_vars=MONTHS,
        var_name="Bulan",
        value_name="Penerima"
    ).assign(
        Bulan=lambda x: pd.Categorical(
            x["Bulan"],
            categories=MONTHS,
            ordered=True
        )
    )


# =========================================================
# SUMBER DATA EXCEL
# =========================================================

with st.sidebar:
    st.header("Data sumber")

    st.caption(
        "Unggah file Excel masing-masing program. "
        "Jika tidak diunggah, aplikasi akan mencari "
        "file Excel di folder yang sama dengan app.py."
    )

    bpnt_upload = st.file_uploader(
        "File BPNT/Sembako (.xlsx)",
        type=["xlsx"],
        key="bpnt_file"
    )

    pkh_upload = st.file_uploader(
        "File PKH (.xlsx)",
        type=["xlsx"],
        key="pkh_file"
    )

    st.divider()

    st.caption(
        "Format yang diharapkan: kolom kecamatan/kelurahan "
        "dan 12 kolom bulan Januari–Desember."
    )


base_dir = Path(__file__).resolve().parent

bpnt_path = base_dir / "bpnt(1).xlsx"
pkh_path = base_dir / "pkh(1).xlsx"


def source_for(upload, path):
    if upload is not None:
        return io.BytesIO(upload.getvalue())

    if path.exists():
        return path

    return None


bpnt_source = source_for(bpnt_upload, bpnt_path)
pkh_source = source_for(pkh_upload, pkh_path)


if bpnt_source is None or pkh_source is None:
    st.warning(
        "Unggah kedua file Excel di panel kiri, atau simpan "
        "file bernama `bpnt(1).xlsx` dan `pkh(1).xlsx` "
        "di folder yang sama dengan `app.py`."
    )
    st.stop()


try:
    bpnt, bpnt_check, bpnt_raw = parse_workbook(
        bpnt_source, "BPNT/Sembako"
    )

    pkh, pkh_check, pkh_raw = parse_workbook(
        pkh_source, "PKH"
    )

except Exception as exc:
    st.error(f"File belum dapat dibaca: {exc}")

    st.info(
        "Periksa nama kolom bulan dan pastikan "
        "file menggunakan format Excel .xlsx."
    )

    st.stop()


# =========================================================
# GABUNG DATA
# =========================================================

all_data = pd.concat(
    [bpnt, pkh],
    ignore_index=True
)

all_check = pd.concat(
    [bpnt_check, pkh_check],
    ignore_index=True
)

long_all = to_long(all_data)


# =========================================================
# PEMERIKSAAN KUALITAS DATA
# =========================================================

quality_flags = []

for program, df in [
    ("BPNT/Sembako", bpnt),
    ("PKH", pkh)
]:

    # Memeriksa nama kelurahan ganda
    if df["Kelurahan"].duplicated().any():
        dupes = df.loc[
            df["Kelurahan"].duplicated(keep=False),
            "Kelurahan"
        ].unique().tolist()

        quality_flags.append(
            f"{program}: nama kelurahan berulang: "
            f"{', '.join(dupes)}."
        )

    # Memeriksa penurunan tajam yang kemudian pulih
    for _, r in df.iterrows():

        vals = r[MONTHS].astype(float).to_numpy()

        for i in range(1, len(vals) - 1):

            future_end = min(len(vals), i + 4)

            future_values = vals[i + 1:future_end]

            if len(future_values) == 0:
                continue

            future_max = np.max(future_values)

            if (
                vals[i - 1] > 0
                and vals[i] < vals[i - 1] * 0.5
                and future_max > vals[i] * 1.8
            ):

                recovery_index = int(
                    np.argmax(future_values)
                )

                recovery_month = MONTHS[
                    i + 1 + recovery_index
                ]

                quality_flags.append(
                    f"{program}: {r['Kelurahan']} "
                    f"turun tajam pada {MONTHS[i]} "
                    f"({format_int(vals[i - 1])} → "
                    f"{format_int(vals[i])}) lalu naik "
                    f"kembali pada {recovery_month} "
                    f"({format_int(future_max)}); "
                    "mohon verifikasi dengan sumber asli."
                )


# =========================================================
# FILTER PROGRAM
# =========================================================

program_options = [
    "Keduanya",
    "BPNT/Sembako",
    "PKH"
]

selected_program = st.selectbox(
    "Program bantuan",
    program_options,
    index=0
)

if selected_program == "Keduanya":
    filtered = all_data.copy()
else:
    filtered = all_data[
        all_data["Program"] == selected_program
    ].copy()


current_month = st.selectbox(
    "Bulan untuk ringkasan wilayah",
    MONTHS,
    index=11
)


# =========================================================
# KARTU RINGKASAN
# =========================================================

jan_total = float(filtered["Januari"].sum())
dec_total = float(filtered["Desember"].sum())

delta = dec_total - jan_total

delta_pct = (
    delta / jan_total * 100
    if jan_total
    else np.nan
)

if selected_program == "Keduanya":

    city_monthly = (
        long_all.groupby(
            "Bulan",
            observed=False
        )["Penerima"].sum()
    )

else:

    city_monthly = (
        long_all[
            long_all["Program"] == selected_program
        ]
        .groupby(
            "Bulan",
            observed=False
        )["Penerima"]
        .sum()
    )

peak_month = city_monthly.idxmax()


k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Penerima Januari",
    format_int(jan_total)
)

k2.metric(
    "Penerima Desember",
    format_int(dec_total),
    delta=(
        f"{format_int(delta)} "
        f"({format_pct(delta_pct)})"
    )
)

k3.metric(
    "Kelurahan tercakup",
    f"{filtered['Kelurahan'].nunique()}"
)

k4.metric(
    "Bulan dengan jumlah tertinggi",
    str(peak_month)
)

st.caption(
    "Angka agregat dashboard dijumlahkan dari baris kelurahan. "
    "Baris subtotal kecamatan dan Total pada file tidak "
    "dijumlahkan ulang."
)


# =========================================================
# TAB DASHBOARD
# =========================================================

(
    tab_overview,
    tab_trend,
    tab_region,
    tab_compare,
    tab_cluster,
    tab_reg,
    tab_quality
) = st.tabs([
    "Ringkasan",
    "Tren bulanan",
    "Sebaran wilayah",
    "Bandingkan program",
    "Klaster K-Means",
    "Regresi eksploratif",
    "Data & validasi"
])


# =========================================================
# TAB 1: RINGKASAN
# =========================================================

with tab_overview:

    st.subheader("Gambaran umum penerima bantuan")

    monthly = long_all.copy()

    if selected_program != "Keduanya":
        monthly = monthly[
            monthly["Program"] == selected_program
        ]

    monthly_sum = (
        monthly.groupby(
            ["Bulan", "Program"],
            observed=False,
            as_index=False
        )["Penerima"].sum()
    )

    fig = px.line(
        monthly_sum,
        x="Bulan",
        y="Penerima",
        color="Program",
        markers=True,
        category_orders={"Bulan": MONTHS},
        title="Perkembangan jumlah penerima per bulan"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend_title_text="Program",
        xaxis_title="Bulan",
        yaxis_title="Jumlah penerima"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            "**Ringkasan perubahan Januari–Desember**"
        )

        rows = []

        for program, df in [
            ("BPNT/Sembako", bpnt),
            ("PKH", pkh)
        ]:

            a = df["Januari"].sum()
            b = df["Desember"].sum()

            rows.append({
                "Program": program,
                "Januari": a,
                "Desember": b,
                "Perubahan": b - a,
                "Perubahan (%)": (
                    (b - a) / a * 100
                    if a else np.nan
                )
            })

        summary = pd.DataFrame(rows)

        st.dataframe(
            summary.style.format({
                "Januari": "{:,.0f}",
                "Desember": "{:,.0f}",
                "Perubahan": "{:+,.0f}",
                "Perubahan (%)": "{:+.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown("**Cara membaca**")

        st.write(
            "Grafik menunjukkan perubahan jumlah penerima "
            "yang tercatat pada setiap bulan. Kenaikan atau "
            "penurunan jumlah penerima tidak dengan sendirinya "
            "menjelaskan penyebab perubahan."
        )

        st.write(
            "Data ini belum memuat jumlah penduduk atau rumah "
            "tangga, sehingga proporsi penerima terhadap "
            "populasi belum dapat dihitung."
        )


# =========================================================
# TAB 2: TREN BULANAN
# =========================================================

with tab_trend:

    st.subheader("Tren bulanan per program")

    trend_data = long_all.copy()

    if selected_program != "Keduanya":
        trend_data = trend_data[
            trend_data["Program"] == selected_program
        ]

    unit = st.radio(
        "Tampilkan tren",
        [
            "Total kota",
            "Per kecamatan",
            "Per kelurahan"
        ],
        horizontal=True
    )

    if unit == "Total kota":

        trend = (
            trend_data.groupby(
                ["Bulan", "Program"],
                observed=False,
                as_index=False
            )["Penerima"].sum()
        )

        fig = px.line(
            trend,
            x="Bulan",
            y="Penerima",
            color="Program",
            markers=True,
            category_orders={"Bulan": MONTHS}
        )

    elif unit == "Per kecamatan":

        trend = (
            trend_data.groupby(
                ["Bulan", "Program", "Kecamatan"],
                observed=False,
                as_index=False
            )["Penerima"].sum()
        )

        fig = px.line(
            trend,
            x="Bulan",
            y="Penerima",
            color="Kecamatan",
            line_dash="Program",
            markers=True,
            category_orders={"Bulan": MONTHS}
        )

    else:

        selected_district = st.selectbox(
            "Pilih kecamatan",
            sorted(filtered["Kecamatan"].unique()),
            key="trend_district"
        )

        trend = trend_data[
            trend_data["Kecamatan"] == selected_district
        ]

        fig = px.line(
            trend,
            x="Bulan",
            y="Penerima",
            color="Kelurahan",
            line_dash="Program",
            markers=True,
            category_orders={"Bulan": MONTHS}
        )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Bulan",
        yaxis_title="Jumlah penerima"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Tren menunjukkan angka yang tercatat pada setiap bulan, "
        "bukan jumlah penerima unik sepanjang tahun."
    )


# =========================================================
# TAB 3: SEBARAN WILAYAH
# =========================================================

with tab_region:

    st.subheader(
        "Sebaran penerima menurut kecamatan dan kelurahan"
        f" — {current_month}"
    )

    region = (
        filtered.groupby(
            ["Kecamatan", "Kelurahan"],
            as_index=False
        )[current_month]
        .sum()
        .rename(columns={current_month: "Penerima"})
    )

    region = region.sort_values(
        "Penerima",
        ascending=False
    )

    district = (
        region.groupby(
            "Kecamatan",
            as_index=False
        )["Penerima"]
        .sum()
        .sort_values("Penerima", ascending=False)
    )

    col_a, col_b = st.columns(2)

    with col_a:

        st.markdown("**Total per kecamatan**")

        fig = px.bar(
            district,
            x="Penerima",
            y="Kecamatan",
            orientation="h",
            text="Penerima",
            color="Penerima",
            color_continuous_scale="Blues"
        )

        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            plot_bgcolor="white",
            paper_bgcolor="white",
            coloraxis_showscale=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col_b:

        st.markdown(
            "**Kelurahan dengan penerima tertinggi "
            "dan terendah**"
        )

        if not region.empty:

            highest = region.iloc[0]
            lowest = region.sort_values(
                "Penerima"
            ).iloc[0]

            st.metric(
                "Tertinggi",
                str(highest["Kelurahan"]),
                format_int(highest["Penerima"])
            )

            st.metric(
                "Terendah",
                str(lowest["Kelurahan"]),
                format_int(lowest["Penerima"])
            )

        st.dataframe(
            region,
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        "Unduh tabel sebaran CSV",
        region.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"sebaran_{current_month.lower()}.csv",
        mime="text/csv"
    )


# =========================================================
# TAB 4: PERBANDINGAN PROGRAM
# =========================================================

with tab_compare:

    st.subheader(
        "Perbandingan pola PKH dan BPNT/Sembako"
    )

    pkh_long = to_long(pkh)
    bpnt_long = to_long(bpnt)

    totals = (
        pd.concat([pkh_long, bpnt_long])
        .groupby(
            ["Program", "Bulan"],
            observed=False,
            as_index=False
        )["Penerima"].sum()
    )

    fig = px.line(
        totals,
        x="Bulan",
        y="Penerima",
        color="Program",
        markers=True,
        category_orders={"Bulan": MONTHS},
        title="Total bulanan kedua program"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    pivot = (
        all_data.groupby(
            ["Kecamatan", "Kelurahan", "Program"],
            as_index=False
        )[["Januari", "Desember"]]
        .sum()
    )

    wide = pivot.pivot(
        index=["Kecamatan", "Kelurahan"],
        columns="Program",
        values=["Januari", "Desember"]
    ).fillna(0)

    wide.columns = [
        f"{a} - {b}" for a, b in wide.columns
    ]

    wide = wide.reset_index()

    if (
        "Desember - PKH" in wide.columns
        and "Desember - BPNT/Sembako" in wide.columns
    ):

        wide["Selisih BPNT-PKH (Desember)"] = (
            wide["Desember - BPNT/Sembako"]
            - wide["Desember - PKH"]
        )

    st.markdown(
        "**Perbandingan Januari dan Desember per kelurahan**"
    )

    st.dataframe(
        wide,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Jumlah PKH dan BPNT tidak selalu mewakili rumah "
        "tangga yang berbeda. Penerima kedua program "
        "bisa saja tumpang tindih."
    )


# =========================================================
# TAB 5: KLASTER K-MEANS
# =========================================================

with tab_cluster:

    st.subheader(
        "Pengelompokan kelurahan menggunakan K-Means"
    )

    st.write(
        "Klaster dibuat berdasarkan pola jumlah penerima "
        "selama 12 bulan. Hasil klaster bersifat deskriptif, "
        "bukan kategori resmi tingkat kemiskinan atau "
        "kelayakan bantuan."
    )

    cluster_program = st.selectbox(
        "Program untuk klaster",
        [
            "BPNT/Sembako",
            "PKH",
            "Gabungan PKH + BPNT"
        ],
        index=0
    )

    n_clusters = st.slider(
        "Jumlah klaster",
        min_value=2,
        max_value=5,
        value=3
    )

    if cluster_program == "Gabungan PKH + BPNT":

        features = (
            all_data.groupby(
                ["Kecamatan", "Kelurahan"],
                as_index=False
            )[MONTHS]
            .sum()
        )

    else:

        source_df = (
            bpnt
            if cluster_program == "BPNT/Sembako"
            else pkh
        )

        features = source_df[
            ["Kecamatan", "Kelurahan"] + MONTHS
        ].copy()

    X = features[MONTHS].astype(float)

    if len(features) < n_clusters:

        st.warning(
            "Jumlah kelurahan lebih sedikit daripada "
            "jumlah klaster yang dipilih."
        )

    else:

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        features["Klaster"] = (
            km.fit_predict(X_scaled) + 1
        )

        features["Rata-rata bulanan"] = (
            X.mean(axis=1)
        )

        features["Jumlah Desember"] = (
            features["Desember"]
        )

        means = (
            features.groupby(
                "Klaster",
                as_index=False
            )[MONTHS]
            .mean()
        )

        means["Rata-rata penerima"] = (
            means[MONTHS].mean(axis=1)
        )

        order = means.sort_values(
            "Rata-rata penerima"
        )["Klaster"].tolist()

        label_map = {
            cl: f"Kelompok {idx + 1}"
            for idx, cl in enumerate(order)
        }

        features["Kelompok relatif"] = (
            features["Klaster"].map(label_map)
        )

        fig = px.scatter(
            features,
            x="Januari",
            y="Desember",
            color="Kelompok relatif",
            hover_name="Kelurahan",
            hover_data=[
                "Kecamatan",
                "Rata-rata bulanan"
            ],
            title=(
                "Posisi kelurahan berdasarkan "
                "jumlah penerima Januari dan Desember"
            )
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            features[
                [
                    "Kecamatan",
                    "Kelurahan",
                    "Kelompok relatif",
                    "Rata-rata bulanan",
                    "Januari",
                    "Desember"
                ]
            ].sort_values(
                ["Kelompok relatif", "Rata-rata bulanan"]
            ),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Unduh hasil klaster CSV",
            features.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="hasil_klaster_kmeans.csv",
            mime="text/csv"
        )


# =========================================================
# TAB 6: REGRESI LINEAR EKSPLORATIF
# =========================================================

with tab_reg:

    st.subheader(
        "Regresi linear eksploratif: hubungan "
        "jumlah PKH dan BPNT"
    )

    st.write(
        "Model ini memeriksa hubungan statistik antara "
        "jumlah penerima kedua program pada tingkat "
        "kelurahan. Model tidak membuktikan sebab-akibat."
    )

    month_reg = st.selectbox(
        "Bulan analisis regresi",
        MONTHS,
        index=11,
        key="reg_month"
    )

    reg_df = pkh[
        ["Kecamatan", "Kelurahan", month_reg]
    ].merge(
        bpnt[
            ["Kecamatan", "Kelurahan", month_reg]
        ],
        on=["Kecamatan", "Kelurahan"],
        suffixes=("_PKH", "_BPNT")
    )

    reg_df = reg_df.rename(
        columns={
            f"{month_reg}_PKH": "PKH",
            f"{month_reg}_BPNT": "BPNT/Sembako"
        }
    ).dropna()

    if (
        len(reg_df) >= 3
        and reg_df["BPNT/Sembako"].nunique() > 1
    ):

        model = LinearRegression().fit(
            reg_df[["BPNT/Sembako"]],
            reg_df["PKH"]
        )

        reg_df["Prediksi PKH"] = model.predict(
            reg_df[["BPNT/Sembako"]]
        )

        r2 = model.score(
            reg_df[["BPNT/Sembako"]],
            reg_df["PKH"]
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("R²", f"{r2:.3f}")

        c2.metric(
            "Koefisien BPNT",
            f"{model.coef_[0]:.3f}"
        )

        c3.metric(
            "Jumlah kelurahan",
            str(len(reg_df))
        )

        fig = px.scatter(
            reg_df,
            x="BPNT/Sembako",
            y="PKH",
            hover_name="Kelurahan",
            trendline="ols",
            title=(
                "Hubungan PKH dan BPNT/Sembako — "
                f"{month_reg}"
            )
        )

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "R² menunjukkan proporsi variasi PKH yang "
            "dapat dijelaskan secara linear oleh BPNT "
            "dalam model pada data ini. Hasil perlu "
            "ditafsirkan secara hati-hati karena jumlah "
            "kelurahan terbatas dan variabel sosial "
            "ekonomi tambahan belum tersedia."
        )

        st.dataframe(
            reg_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "Data belum cukup bervariasi untuk menghitung "
            "regresi pada bulan ini."
        )


# =========================================================
# TAB 7: DATA DAN VALIDASI
# =========================================================

with tab_quality:

    st.subheader("Pemeriksaan data dan sumber")

    st.markdown(
        "**Perbandingan total file dengan penjumlahan "
        "baris kelurahan**"
    )

    st.write(
        "Selisih dihitung dari total hasil penjumlahan "
        "kelurahan dikurangi angka pada baris Total "
        "di workbook. Dashboard menggunakan jumlah "
        "baris kelurahan agar subtotal tidak dihitung "
        "dua kali."
    )

    check_view = all_check.copy()

    check_view["Bulan"] = pd.Categorical(
        check_view["Bulan"],
        categories=MONTHS,
        ordered=True
    )

    st.dataframe(
        check_view,
        use_container_width=True,
        hide_index=True
    )

    nonzero = check_view[
        check_view[
            "Selisih (kelurahan - file)"
        ].abs() > 0.001
    ]

    if not nonzero.empty:

        st.warning(
            f"Ditemukan {len(nonzero)} bulan-program "
            "dengan perbedaan antara jumlah baris "
            "kelurahan dan total pada file. Periksa "
            "kembali workbook sumber sebelum menggunakan "
            "angka untuk laporan resmi."
        )

        st.dataframe(
            nonzero,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.success(
            "Total dari baris kelurahan cocok dengan "
            "total pada file untuk semua bulan "
            "yang memiliki total."
        )

    if quality_flags:

        st.markdown(
            "**Pola angka yang perlu diverifikasi**"
        )

        for flag in sorted(set(quality_flags)):
            st.warning(flag)

    else:

        st.success(
            "Tidak ditemukan pola penurunan tajam "
            "yang kemudian pulih berdasarkan "
            "pemeriksaan sederhana."
        )

    st.markdown("**Keterbatasan analisis**")

    st.markdown("""
    - Data jumlah penduduk atau rumah tangga belum tersedia,
      sehingga proporsi penerima belum dapat dihitung.
    - Variabel sosial ekonomi tambahan belum tersedia,
      sehingga penyebab perbedaan antarwilayah belum
      dapat disimpulkan.
    - Tren menunjukkan angka yang tercatat per bulan,
      bukan jumlah penerima unik sepanjang tahun.
    - K-Means mengelompokkan kemiripan pola angka,
      bukan menentukan kelayakan bantuan.
    """)

    with st.expander(
        "Lihat data kelurahan hasil pembacaan"
    ):

        st.dataframe(
            all_data,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Unduh data bersih gabungan CSV",
            all_data.to_csv(
                index=False
            ).encode("utf-8-sig"),
            file_name="data_bansos_tanjungpinang_2025.csv",
            mime="text/csv"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
<p class="small-note">
Dashboard analisis deskriptif |
Sumber data: file Excel PKH dan BPNT/Sembako yang disediakan |
Periksa catatan validasi sebelum mengutip angka dalam laporan.
</p>
""", unsafe_allow_html=True)    