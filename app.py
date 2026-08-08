import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import joblib
import os
from prophet import Prophet
from datetime import datetime

# ════════════════════════════════
# KONFIGURASI
# ════════════════════════════════
st.set_page_config(
    page_title="Sistem Prediksi Kurs", 
    layout="wide",
    initial_sidebar_state="auto"
)

# ── INJEKSI CSS UNTUK RESPONSIVE MOBILE ──
st.markdown("""
    <style>
    /* Mengurangi padding untuk layar kecil (HP) dan merapikan font */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 1.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        h1 {
            font-size: 1.8rem !important;
        }
        h2, h3 {
            font-size: 1.3rem !important;
        }
        .stMetric label {
            font-size: 0.9rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

API_KEY         = "2b91e09366f4199837a35023"
MODEL_PATH      = "models/"
ACTUAL_DATA_PATH = "actual_data/"
MATA_UANG       = ["USD", "SGD", "MYR", "JPY", "CNY", "EUR", "AUD", "SAR"]

MAPE_RESULTS = {
    "USD": 2.52, "SGD": 2.96, "MYR": 8.05, "JPY": 7.18,
    "CNY": 7.85, "EUR": 4.22, "AUD": 6.24, "SAR": 2.50
}


# ════════════════════════════════
# FUNGSI: LOAD DATA & MODEL
# ════════════════════════════════

@st.cache_resource
def load_model(currency):
    path = MODEL_PATH + f"prophet_{currency}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data(ttl=3600)
def get_realtime(currency):
    try:
        url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/{currency}"
        r = requests.get(url, timeout=10)
        return r.json()["conversion_rates"]["IDR"]
    except:
        return None

@st.cache_data
def load_actual_data(currency):
    path = ACTUAL_DATA_PATH + f"Januari_{currency}.xlsx"
    if not os.path.exists(path):
        st.error(f"DEBUG: File tidak ditemukan di path: {os.path.abspath(path)}")
        return None
    try:
        df = pd.read_excel(path, sheet_name=0, header=4)
        df["ds"] = pd.to_datetime(df["Tanggal"])
        df["y_actual"] = df["Kurs Jual"] / df["Nilai"]
        df = df[(df["ds"] >= "2026-01-01") & (df["ds"] <= "2026-01-09")]
        return df[["ds", "y_actual"]].sort_values("ds").reset_index(drop=True)
    except Exception as e:
        st.error(f"DEBUG: Error saat baca file: {e}")
        return None


# ════════════════════════════════
# FUNGSI: FORECASTING
# ════════════════════════════════

def get_forecast_and_history(model):
    history = model.history.copy()
    history["y"] = np.exp(history["y"])

    future = model.make_future_dataframe(periods=7, freq='B')
    forecast = model.predict(future)
    forecast["yhat"]       = np.exp(forecast["yhat"])
    forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
    forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

    forecast_only = forecast.tail(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)
    return history, forecast, forecast_only


# ════════════════════════════════
# FUNGSI: KOMPONEN INSIGHT & INDIKATOR
# ════════════════════════════════

def get_trend_indicator(forecast_df):
    first_val = forecast_df["yhat"].iloc[0]
    last_val  = forecast_df["yhat"].iloc[-1]
    pct_change = ((last_val - first_val) / first_val) * 100

    if pct_change > 0.3:
        return "📈", "Tren Naik", f"+{pct_change:.2f}%", "#22c55e"
    elif pct_change < -0.3:
        return "📉", "Tren Turun", f"{pct_change:.2f}%", "#ef4444"
    else:
        return "➡️", "Stabil", f"{pct_change:+.2f}%", "#f59e0b"


def generate_forecast_insight(selected, forecast_df, mape_val):
    first_val  = forecast_df["yhat"].iloc[0]
    last_val   = forecast_df["yhat"].iloc[-1]
    min_pred   = forecast_df["yhat_lower"].min()
    max_pred   = forecast_df["yhat_upper"].max()
    pct_change = ((last_val - first_val) / first_val) * 100
    avg_pred   = forecast_df["yhat"].mean()

    start_date = forecast_df["ds"].iloc[0].strftime("%d %b %Y")
    end_date   = forecast_df["ds"].iloc[-1].strftime("%d %b %Y")

    insights = []

    if abs(pct_change) <= 0.3:
        insights.append(
            f"🔹 Pada periode pengujian ({start_date} – {end_date}), prediksi menunjukkan **{selected}/IDR cenderung stabil** "
            f"dengan perubahan hanya **{pct_change:+.2f}%**."
        )
    elif pct_change > 0:
        insights.append(
            f"🔹 Pada periode pengujian ({start_date} – {end_date}), prediksi menunjukkan **{selected}/IDR cenderung menguat {pct_change:.2f}%** "
            f"(IDR melemah terhadap {selected})."
        )
    else:
        insights.append(
            f"🔹 Pada periode pengujian ({start_date} – {end_date}), prediksi menunjukkan **{selected}/IDR cenderung melemah {abs(pct_change):.2f}%** "
            f"(IDR menguat terhadap {selected})."
        )

    insights.append(
        f"🔹 Tren prediksi bergerak dalam rentang **Rp{min_pred:,.0f} – Rp{max_pred:,.0f}**, "
        f"dengan rata-rata prediksi sebesar **Rp{avg_pred:,.2f}**."
    )

    insights.append(
        f"🔹 Berdasarkan pengujian, model Prophet terbukti sangat akurat dengan nilai metrik kesalahan (MAPE) "
        f"hanya sebesar **{mape_val}%** terhadap data aktual Bank Indonesia pada periode yang sama."
    )

    return insights


# ════════════════════════════════
# FUNGSI: UTILITAS
# ════════════════════════════════

def format_wib():
    BULAN = {
        1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
        5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
        9: "September", 10: "Oktober", 11: "November", 12: "Desember"
    }
    from datetime import timezone, timedelta
    wib = timezone(timedelta(hours=7))
    now_wib = datetime.now(wib)
    return f"{now_wib.day} {BULAN[now_wib.month]} {now_wib.year}, {now_wib.strftime('%H:%M')} WIB"


# ════════════════════════════════════════════════════════════════
# UI — HEADER
# ════════════════════════════════════════════════════════════════

st.title("💱 Sistem Prediksi Kurs Mata Uang")
st.markdown(
    """
    <div style='text-align: justify; color: rgba(250,250,250,0.8); font-size: 0.9rem; line-height: 1.6; margin-bottom: 1.5rem;'>
        Sistem Prediksi Kurs Mata Uang ini menggabungkan data historis Bank Indonesia, kurs realtime dari
        ExchangeRate-API, serta teknologi forecasting menggunakan algoritma Prophet. Pengguna dapat memantau
        pergerakan kurs dari 8 mata uang utama terhadap Rupiah, melihat nilai tukar terkini, serta memperoleh
        prediksi kurs beberapa hari ke depan yang telah dievaluasi menggunakan metode MAPE untuk mengukur
        tingkat akurasi model.
    </div>
    """,
    unsafe_allow_html=True
)

selected = st.selectbox("Pilih Mata Uang", MATA_UANG, label_visibility="visible")
st.divider()


# ════════════════════════════════════════════════════════════════
# UI — DETAIL NILAI TUKAR & KONVERTER
# ════════════════════════════════════════════════════════════════

st.subheader("🔍 Detail Nilai Tukar Mata Uang")

col1, col2 = st.columns(2)
realtime = get_realtime(selected)

with col1:
    st.metric(
        label=f"Kurs Realtime {selected} → IDR",
        value=f"Rp {realtime:,.2f}" if realtime else "Tidak tersedia"
    )

with col2:
    mape_val = MAPE_RESULTS.get(selected, "-")
    st.metric(
        label="MAPE Model",
        value=f"{mape_val}%"
    )

st.markdown("**Konverter Kurs**")

if "swap" not in st.session_state:
    st.session_state.swap = False

if st.session_state.swap:
    from_curr = selected
    to_curr = "IDR"
else:
    from_curr = "IDR"
    to_curr = selected

col1, col2, col3 = st.columns([3, 1, 2])

with col1:
    jumlah = st.number_input(
        "Jumlah",
        min_value=0,
        value=1,
        step=1,
        format="%d"
    )

with col2:
    st.markdown(
        """
        <div style='font-size:14px; line-height:1.6; margin-bottom:0.4rem; color:rgba(250,250,250,0.8); text-align: center;'>
            konversi
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("⇅", use_container_width=True):
        st.session_state.swap = not st.session_state.swap
        st.rerun()

with col3:
    st.markdown(
        f"""
        <div style='font-size:20px; font-weight:bold; margin-top:2rem; text-align: center;'>
            {from_curr} → {to_curr}
        </div>
        """,
        unsafe_allow_html=True
    )

if realtime:
    if from_curr == "IDR":
        hasil = jumlah / realtime
        st.success(f"**{jumlah:,} IDR = {hasil:,.6f} {selected}**")
    else:
        hasil = jumlah * realtime
        st.success(f"**{jumlah:,} {selected} = Rp {hasil:,.2f}**")
    st.caption(f"Kurs diperbarui: {format_wib()}")
else:
    st.warning("Kurs realtime tidak tersedia, konverter tidak dapat dihitung.")

st.divider()


# ════════════════════════════════════════════════════════════════
# UI — PREDIKSI, VALIDASI, & GRAFIK
# ════════════════════════════════════════════════════════════════

model = load_model(selected)

if model is None:
    st.warning(f"Model untuk {selected} belum tersedia. Pastikan file `models/prophet_{selected}.pkl` ada.")
else:
    history, forecast_full, forecast_df = get_forecast_and_history(model)

    st.subheader("📊 Prediksi (Model Prophet)")

    icon, label_tren, pct_str, warna = get_trend_indicator(forecast_df)

    st.markdown(
        f"""
        <div style='
            display: inline-flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.5rem;
            background-color: {warna}22;
            border: 1px solid {warna};
            border-radius: 8px;
            padding: 0.4rem 1rem;
            margin-bottom: 1rem;
        '>
            <span style='font-size: 1.2rem'>{icon}</span>
            <span style='color: {warna}; font-weight: 700; font-size: 0.95rem'>{label_tren}</span>
            <span style='color: {warna}; font-size: 0.85rem'>({pct_str} dalam 7 hari)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**Tabel Prediksi 7 Hari Ke Depan**")
    display_df = forecast_df.copy()
    display_df.columns = ["Tanggal", "Prediksi (IDR)", "Batas Bawah", "Batas Atas"]
    display_df["Tanggal"] = display_df["Tanggal"].dt.strftime("%d %b %Y")
    for col in ["Prediksi (IDR)", "Batas Bawah", "Batas Atas"]:
        display_df[col] = display_df[col].apply(lambda x: f"Rp {x:,.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("📈 Grafik Prediksi (7 Hari Ke Depan)")
    future_dates = forecast_full["ds"].tail(7)
    future_pred  = forecast_full["yhat"].tail(7)
    future_lower = forecast_full["yhat_lower"].tail(7)
    future_upper = forecast_full["yhat_upper"].tail(7)

    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=pd.concat([future_dates, future_dates[::-1]]),
            y=pd.concat([future_upper, future_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(255,165,0,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="Confidence Interval"
        )
    )
    fig1.add_trace(# pyright: ignore
        go.Scatter(
            x=future_dates,
            y=future_pred,
            mode="lines+markers",
            name="Prediksi",
            line=dict(color="#F59E0B", width=3)
        )
    )
    if realtime:
        fig1.add_hline(
            y=realtime,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Realtime: Rp {realtime:,.0f}"
        )
    fig1.update_layout(
        title=f"Prediksi {selected}/IDR",
        xaxis_title="Tanggal",
        yaxis_title="Kurs (IDR)",
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10) # Sesuaikan margin untuk mobile
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("**💡 Insight Prediksi**")
    mape_value = MAPE_RESULTS[selected]
    insights = generate_forecast_insight(selected, forecast_df, mape_value)
    for insight in insights:
        st.markdown(insight)
    
    st.divider()

    st.subheader("✅ Validasi: Aktual (Bank Indonesia) vs Prediksi")

    actual_df = load_actual_data(selected)

    if actual_df is not None and not actual_df.empty:
        validasi_df = forecast_df.merge(actual_df, on="ds", how="inner")

        if not validasi_df.empty:
            fig_val = go.Figure()
            fig_val.add_trace(
                go.Scatter(
                    x=validasi_df["ds"], y=validasi_df["y_actual"],
                    mode="lines+markers", name="Aktual (BI)",
                    line=dict(color="#4F46E5", width=3)
                )
            )
            fig_val.add_trace(
                go.Scatter(
                    x=validasi_df["ds"], y=validasi_df["yhat"],
                    mode="lines+markers", name="Prediksi (Prophet)",
                    line=dict(color="#F59E0B", width=3, dash="dash")
                )
            )
            fig_val.update_layout(
                title=f"Aktual vs Prediksi {selected}/IDR (1–9 Jan 2026)",
                xaxis_title="Tanggal",
                yaxis_title="Kurs (IDR)",
                template="plotly_dark",
                height=450,
                hovermode="x unified",
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_val, use_container_width=True)
            
            st.caption("ℹ️: Hari libur bursa/Bank Indonesia otomatis dilewati pada grafik validasi.")
    st.divider()

    st.subheader("📈 Grafik Historis (2015 - 2025)")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=history["ds"],
            y=history["y"],
            mode="lines",
            name="Historis",
            line=dict(color="#4F46E5", width=2)
        )
    )
    if realtime:
        fig2.add_hline(
            y=realtime,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Realtime: Rp {realtime:,.0f}"
        )
    fig2.update_layout(
        title=f"Historis {selected}/IDR",
        xaxis_title="Tanggal",
        yaxis_title="Kurs (IDR)",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.divider()

    with st.expander("Informasi Model", expanded=False):
        jumlah_data = len(history)
        tgl_awal    = history["ds"].min().strftime("%Y")
        tgl_akhir   = history["ds"].max().strftime("%Y")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.markdown("<div style='text-align:center; font-size:0.9rem;'><b>Algoritma</b></div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center; font-size:0.9rem;'>Prophet</div>", unsafe_allow_html=True)
        with col_m2:
            st.markdown("<div style='text-align:center; font-size:0.9rem;'><b>Periode</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; font-size:0.9rem;'>{tgl_awal}–{tgl_akhir}</div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown("<div style='text-align:center; font-size:0.9rem;'><b>Observasi</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; font-size:0.9rem;'>{jumlah_data:,}</div>", unsafe_allow_html=True)
 
        st.markdown("---")
        st.markdown(
        """
        <div style='text-align: justify; line-height: 1.6; font-size: 0.9rem; margin-bottom: 1rem;'>
            Model prediksi dikembangkan menggunakan algoritma Prophet yang dilatih menggunakan data historis nilai tukar mata uang dari Bank Indonesia periode 2015–2025. Model menghasilkan prediksi kurs untuk 7 hari ke depan. Keandalan hasil prediksi divalidasi melalui perbandingan dengan data aktual Bank Indonesia.
        </div>
        """,
            unsafe_allow_html=True
        )
        