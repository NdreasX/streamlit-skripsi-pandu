# ============================================================================
# CAMPAIGN NAVIGATOR: BUDGET ROI OPTIMIZER v4.1 (ENTERPRISE EDITION)
# Features: Recursive Forecast, Smart Absorption, Bar Charts, Auto-Insights & Tooltips
# ============================================================================

# ============================================================================
# IMPORTS
# ============================================================================

import os
import pickle
import datetime
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Campaign Navigator v4.1",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS buat mempercantik metrik dan alert
st.markdown("""
    <style>
    .stMetric { background-color: #1E1E1E; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;}
    .alert-box { padding: 15px; border-radius: 8px; margin-top: 10px; margin-bottom: 20px;}
    .alert-warning { background-color: #ff980020; border-left: 5px solid #ff9800; color: #ff9800;}
    .alert-danger { background-color: #f4433620; border-left: 5px solid #f44336; color: #f44336;}
    .alert-success { background-color: #4caf5020; border-left: 5px solid #4caf50; color: #4caf50;}
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONSTANTS
# ============================================================================

DATA_PATH = "" 
FUNNELS = ["awareness", "conversion", "engagement", "session"] 

HORIZONS = [1, 3, 7]
DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

SCENARIO_BUDGETS = {
    "Sangat Konservatif (-30%)": -30,
    "Konservatif (-15%)": -15,
    "Baseline (0%)": 0,
    "Optimis (+15%)": +15,
    "Sangat Agresif (+30%)": +30
}

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

@st.cache_data
def load_eda_data():
    df = pd.read_csv(f"{DATA_PATH}grouped_df_agregate.csv")
    df["day"] = pd.to_datetime(df["day"])
    df = df.sort_values(['ad_set', 'day']).reset_index(drop=True)
    return df

@st.cache_data
def load_deployment_metadata():
    stats = joblib.load(f"{DATA_PATH}dataset_stats.pkl")
    feature_cols = joblib.load(f"{DATA_PATH}feature_columns.pkl")
    label_encoders = joblib.load(f"{DATA_PATH}label_encoders.pkl")
    deployment_df = pd.read_csv(f"{DATA_PATH}deployment_models_df.csv")
    return {
        'stats': stats, 'feature_cols': feature_cols, 'encoders': label_encoders,
        'deployment_df': deployment_df
    }

@st.cache_resource
def load_model(metric: str, horizon: int):
    filepath = f"{DATA_PATH}{metric}_t{horizon}_rf.pkl"
    model_obj = joblib.load(filepath)
    return model_obj["model"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_daily_aggregates(df):
    return df.groupby("day").agg({"CTR": "mean", "CPC": "mean", "clicks": "sum", "spend": "sum"}).reset_index()

def create_day_of_week_metrics(df):
    df = df.copy()
    df["day_of_week_name"] = pd.to_datetime(df["day"]).dt.day_name()
    dow_ctr = df.groupby("day_of_week_name")["CTR"].mean().reindex(DAYS_ORDER).reset_index()
    dow_cpc = df.groupby("day_of_week_name")["CPC"].mean().reindex(DAYS_ORDER).reset_index()
    dow_clicks = df.groupby("day_of_week_name")["clicks"].mean().reindex(DAYS_ORDER).reset_index()
    return dow_ctr, dow_cpc, dow_clicks

def build_input_features(budget, funnel, start_date, stats, encoders, feature_cols, ctr_lag_1=0.0, cpc_lag_1=0.0, clicks_lag_1=0.0):
    date = pd.to_datetime(start_date)
    row = {
        "impressions": budget * stats["impressions_ratio"],
        "reach": budget * stats["reach_ratio"],
        "spend": budget,
        "brand_enc": encoders["brand"].transform([stats["brand_mode"]])[0],
        "cta_enc": encoders["cta"].transform([stats["cta_mode"]])[0],
        "funnel_enc": encoders["funnel"].transform([funnel])[0],
        "month": date.month,
        "quarter": date.quarter,
        "week": date.isocalendar().week,
        "day_of_week": date.dayofweek,
        "is_weekend": int(date.dayofweek >= 5),
        "holiday": 0,
        "CTR_lag_1": ctr_lag_1,
        "CPC_lag_1": cpc_lag_1,
        "clicks_lag_1": clicks_lag_1,
    }
    for col in feature_cols:
        if col not in row: row[col] = 0
    return pd.DataFrame([row])[feature_cols]

def generate_forecast_recursive(budget, funnel, start_date, horizon, metadata, historical_df):
    start_ts = pd.to_datetime(start_date)
    forecast_dates = pd.date_range(start=start_ts, periods=horizon)
    
    ctr_model = load_model("ctr", horizon)
    cpc_model = load_model("cpc", horizon)
    
    preds_ctr, preds_cpc, preds_clicks = [], [], []
    hist_funnel = historical_df[historical_df['funnel'] == funnel].sort_values('day')
    
    # Rata-rata khusus funnel sebagai patokan ML
    if not hist_funnel.empty:
        hist_ctr_base = hist_funnel['CTR'].mean()
        hist_cpc_base = hist_funnel['CPC'].mean()
    else:
        hist_ctr_base = metadata['stats'].get('ctr_mean', 0.008)
        hist_cpc_base = metadata['stats'].get('cpc_mean', 700)
        
    lag_ctr, lag_cpc, lag_clicks = hist_ctr_base, hist_cpc_base, 0.0

    for i in range(horizon):
        current_date = start_ts + pd.Timedelta(days=i)
        X_input = build_input_features(budget, funnel, current_date, metadata['stats'], metadata['encoders'], metadata['feature_cols'], lag_ctr, lag_cpc, lag_clicks)
        
        raw_ctr = max(0.0001, ctr_model.predict(X_input)[0])
        raw_cpc = max(10.0, cpc_model.predict(X_input)[0])
        
        # Fluktuasi Kalender Dinamis
        dow = current_date.dayofweek
        dow_mult_ctr = {0: 0.95, 1: 0.98, 2: 1.0, 3: 1.0, 4: 1.02, 5: 1.08, 6: 1.05}
        dow_mult_cpc = {0: 0.98, 1: 0.98, 2: 1.0, 3: 1.0, 4: 1.02, 5: 1.05, 6: 1.03}
        
        raw_ctr *= dow_mult_ctr[dow]
        raw_cpc *= dow_mult_cpc[dow]
        
        # Rem Kejenuhan Budget (Diminishing Returns)
        THRESHOLD_BUDGET = 5000000.0  
        if budget > THRESHOLD_BUDGET:
            penalty_factor = (budget / THRESHOLD_BUDGET) ** 0.2
            pred_ctr = raw_ctr / penalty_factor
            pred_cpc = raw_cpc * penalty_factor
        else:
            pred_ctr = raw_ctr
            pred_cpc = raw_cpc
            
        # Bypass Limit ML & Hitung Clicks secara Matematis
        absorption_rate = (pred_ctr / hist_ctr_base) * 0.85
        absorption_rate = min(1.05, max(0.3, absorption_rate))
        
        pred_spend = budget * absorption_rate
        pred_clicks = pred_spend / pred_cpc if pred_cpc > 0 else 0
        
        preds_ctr.append(pred_ctr)
        preds_cpc.append(pred_cpc)
        preds_clicks.append(pred_clicks)
        
        lag_ctr, lag_cpc, lag_clicks = pred_ctr, pred_cpc, pred_clicks
        
    return pd.DataFrame({"Date": forecast_dates, "CTR": preds_ctr, "CPC": preds_cpc, "Clicks": preds_clicks})

def run_scenario_simulations(budget_baseline, funnel, start_date, horizon, metadata, scenario_adjustments, historical_df):
    scenarios = {}
    for scenario_name, adjustment_pct in scenario_adjustments.items():
        budget_adjusted = budget_baseline * (1 + adjustment_pct / 100)
        forecast = generate_forecast_recursive(budget_adjusted, funnel, start_date, horizon, metadata, historical_df)
        forecast['Scenario'] = scenario_name
        forecast['Budget Adjustment'] = f"{adjustment_pct:+.0f}%"
        forecast['Budget'] = f"Rp {budget_adjusted:,.0f}"
        scenarios[scenario_name] = forecast
    return scenarios

def plot_with_benchmark(df, x_col, y_col, title, benchmark_val, higher_is_better=True):
    fig = go.Figure()
    formatted_dates = pd.to_datetime(df[x_col]).dt.strftime('%d %b')
    
    # Logika 3 Warna Lampu Lalu Lintas
    colors = []
    for val in df[y_col]:
        if higher_is_better: 
            if val < benchmark_val * 0.8: colors.append('#F44336')
            elif val > benchmark_val * 1.1: colors.append('#4CAF50')
            else: colors.append('#2196F3')
        else: 
            if val > benchmark_val * 1.2: colors.append('#F44336')
            elif val < benchmark_val * 0.9: colors.append('#4CAF50')
            else: colors.append('#2196F3')
            
    fig.add_trace(go.Bar(
        x=formatted_dates, y=df[y_col], name=f'Prediksi {y_col}', marker_color=colors,
        text=df[y_col], texttemplate='%{text:,.0f}' if y_col != 'CTR' else '%{text:.4f}', textposition='auto'
    ))
    
    fig.add_hline(
        y=benchmark_val, line_dash="dash", line_color="#FFC107", 
        annotation_text="Rata-rata Historis", annotation_position="top left"
    )
    
    fig.update_layout(
        title=title, template="plotly_dark", margin=dict(l=0, r=0, t=40, b=0),
        xaxis=dict(type='category', showgrid=False), yaxis=dict(showgrid=True, gridcolor='#444'),
        showlegend=False
    )
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

eda_df = load_eda_data()
metadata = load_deployment_metadata()

# Header Utama & Onboarding
st.title("🎯 Campaign Navigator")
st.markdown("**Aplikasi Cerdas Penentu Estimasi Budget & Performa Meta Ads**")

with st.expander("💡 Baca Dulu: Cara Pakai & Gimana Mesin Ini Bekerja"):
    st.markdown("""
    **Apa sih aplikasi ini?**
    Aplikasi ini ngebantu lo nebak (forecasting) performa iklan Meta Ads lo di masa depan. Daripada lo nebak-nebak buah manggis atau bakar duit sembarangan, mesin AI kita bakal ngasih estimasi **Berapa Klik yang lo dapet**, **Berapa Harga per Klik (CPC)**, dan **Berapa Total Budget yang bakal kesedot (Spend)**.

    **Cara Kerja Mesinnya:**
    Sistem ini belajar dari data historis iklan masa lalu skala Enterprise (Retail Elektronik Besar). Untuk mendapatkan performa optimal (Hijau), budget harian disarankan menyesuaikan skala tersebut (di atas Rp 2 Juta per hari) agar menang lelang audiens menengah ke atas.
    """)

st.divider()

st.subheader("✅ Persetujuan Pengguna")
agree = st.checkbox("Gue paham kalau aplikasi ini ngasih angka estimasi (perkiraan) dari data masa lalu, bukan ramalan masa depan yang 100% pasti.")
if not agree:
    st.warning("Yuk, centang dulu kotak di atas biar bisa mulai simulasinya!")
    st.stop()
st.divider()

tab_forecast, tab_eda, tab_info = st.tabs(["🔮 Simulasi Budget (Forecaster)", "📊 Data Historis", "🧠 Info Sistem"])

with tab_forecast:
    st.header("🚀 Pengaturan Iklan")
    col1, col2, col3, col4 = st.columns(4)
    with col1: budget = st.number_input("Baseline Budget (IDR)", min_value=10000, value=1000000, step=50000)
    with col2: funnel = st.selectbox("Target Funnel", FUNNELS)
    with col3: horizon = st.selectbox("Forecast Horizon", HORIZONS)
    with col4: start_date = st.date_input("Start Date")

    st.subheader("📊 Pilihan Skenario")
    col_scen1, col_scen2 = st.columns(2)
    with col_scen1: run_single = st.checkbox("Jalankan 1 Skenario Saja (Sesuai Budget Di Atas)", value=True)
    with col_scen2:
        if not run_single:
            selected_scenarios = st.multiselect("Pilih Perbandingan Budget:", list(SCENARIO_BUDGETS.keys()), default=list(SCENARIO_BUDGETS.keys()))

    if st.button("🎯 Mulai Prediksi!", type="primary", use_container_width=True):
        st.divider()
        
        # Benchmark Khusus per Funnel
        hist_funnel = eda_df[eda_df['funnel'] == funnel]
        if not hist_funnel.empty:
            hist_cpc = hist_funnel['CPC'].mean()
            hist_ctr = hist_funnel['CTR'].mean()
            hist_clicks = hist_funnel['clicks'].mean()
        else:
            hist_cpc = metadata['stats'].get('cpc_mean', 700)
            hist_ctr = metadata['stats'].get('ctr_mean', 0.008)
            hist_clicks = metadata['stats'].get('clicks_mean', 50)
            
        if run_single:
            st.subheader("📊 Hasil Prediksi (Estimasi Rentang)")
            with st.spinner("Mesin lagi ngitung probabilitas..."):
                forecast_df = generate_forecast_recursive(budget, funnel, start_date, horizon, metadata, eda_df)
                
            mean_ctr, mean_cpc, mean_clicks = forecast_df["CTR"].mean(), forecast_df["CPC"].mean(), forecast_df["Clicks"].mean()
            
            raw_estimated_spend = mean_cpc * mean_clicks
            max_possible_spend = budget * 1.05 
            estimated_spend = min(raw_estimated_spend, max_possible_spend)
            if estimated_spend == max_possible_spend and mean_cpc > 0: mean_clicks = estimated_spend / mean_cpc
                
            ERR_MARGIN = 0.15 
            
            # Tampilan Metrik dengan Format Persen & Penjelasan User-Friendly
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                # CTR dikali 100 biar jadi format persen (Misal: 0.69%)
                st.metric("Est. CTR (Daya Tarik Iklan)", f"{mean_ctr*(1-ERR_MARGIN)*100:.2f}% - {mean_ctr*(1+ERR_MARGIN)*100:.2f}%")
                st.caption("✨ **Porsi orang yang ngeklik** setelah ngelihat iklan lo. Makin gede persentasenya, makin bagus materi iklan lo.")
                
            with c2:
                st.metric("Est. CPC (Harga per Klik)", f"Rp {mean_cpc*(1-ERR_MARGIN):,.0f} - {mean_cpc*(1+ERR_MARGIN):,.0f}")
                st.caption("💸 **Biaya rata-rata** yang lo bayar ke Meta tiap ada 1 orang yang ngeklik. Makin murah harganya, makin hemat budget lo.")
                
            with c3:
                st.metric("Est. Clicks (Total Klik)", f"{mean_clicks*(1-ERR_MARGIN):,.0f} - {mean_clicks*(1+ERR_MARGIN):,.0f}")
                st.caption("🖱️ **Estimasi total pengunjung** yang bakal beneran mampir ke link/website lo dari iklan ini.")
                
            with c4:
                st.metric("Est. Spend (Uang Terpakai)", f"Rp {estimated_spend*(1-ERR_MARGIN):,.0f} - {min(estimated_spend*(1+ERR_MARGIN), max_possible_spend):,.0f}")
                st.caption("💰 **Total uang lo yang bakal disedot** sama Meta. Kadang nggak habis semua kalau iklannya kurang optimal.")

            # Smart Actionable Insights
            spend_ratio = (estimated_spend / budget) * 100 
            st.subheader("💡 Rekomendasi & Strategi")
            if spend_ratio < 70:
                st.markdown(f"""<div class='alert-box alert-warning'>
                <strong>🐢 Iklan Kurang Bensin (Penyerapan Cuma {spend_ratio:.1f}%)</strong><br>
                <strong>🤔 Kenapa bisa gini?</strong> Prediksi interaksi iklan lo (CTR) lebih rendah dari standar atau budget terlalu kecil untuk skala Enterprise. Algoritma Meta 'males' nayangin iklan yang jarang diklik karena kalah lelang.<br>
                <strong>🎯 Saran Action:</strong> Perluas target audiens lo, rombak materi iklan, atau sesuaikan budget harian lo ke standar Enterprise (2-5 Juta).
                </div>""", unsafe_allow_html=True)
            elif 70 <= spend_ratio <= 95:
                st.markdown(f"""<div class='alert-box alert-success'>
                <strong>🎯 Jalan Mulus / Optimal (Penyerapan Ideal {spend_ratio:.1f}%)</strong><br>
                <strong>🤔 Kenapa bisa gini?</strong> Ada keseimbangan sempurna antara besaran budget, audiens, dan daya tarik iklan lo. Meta punya ruang pas buat mendistribusikan iklan lo secara stabil.<br>
                <strong>🎯 Saran Action:</strong> Pantau terus. Kalau yang klik beneran pada beli (ROI bagus), lo siap buat Scale-Up (naikin budget) pelan-pelan.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='alert-box alert-danger'>
                <strong>🔥 Iklan Bocor Alus (Budget Mentok {spend_ratio:.1f}%)</strong><br>
                <strong>🤔 Kenapa bisa gini?</strong> Daya tarik iklan lo luar biasa di audiens yang luas, tapi budget lo mentok! Iklan lo bakal ludes dan mati sebelum sore/malam hari.<br>
                <strong>🎯 Saran Action:</strong> Kalau performanya nguntungin, <strong>gas naikin budget harian lo</strong> biar iklan nggak mati di jam-jam produktif.
                </div>""", unsafe_allow_html=True)

            # Visualizations with Bar Charts and Dynamic Text
            st.subheader(f"📊 Tren Performa ({funnel.title()} Funnel)")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Agar visual klik masuk akal dengan budget, benchmark klik dinaikkan seiring budget
                dynamic_hist_clicks = (budget / 1000000) * hist_clicks if budget > 1000000 else hist_clicks
                st.plotly_chart(plot_with_benchmark(forecast_df, "Date", "Clicks", "Prediksi Traffic (Total Klik Masuk)", dynamic_hist_clicks, True), use_container_width=True)
                
                # Kesimpulan Dinamis Grafik Klik
                if mean_clicks > dynamic_hist_clicks * 1.1:
                    st.success(f"📈 **Membaca Grafik Traffic:** Total klik diprediksi sangat berlimpah! Materi iklan lo berpotensi besar menarik banyak *traffic*!")
                elif mean_clicks < dynamic_hist_clicks * 0.9:
                    st.warning(f"📉 **Membaca Grafik Traffic:** Total klik diprediksi **berada di bawah rata-rata**. Sepertinya audiens kurang tertarik atau budget kekecilan.")
                else:
                    st.info("⚖️ **Membaca Grafik Traffic:** Jumlah klik yang masuk diprediksi **stabil dan wajar** sesuai dengan standar rata-rata data masa lalu.")
                    
            with col_chart2:
                st.plotly_chart(plot_with_benchmark(forecast_df, "Date", "CPC", "Prediksi Harga per Klik (Biaya CPC)", hist_cpc, False), use_container_width=True)
                
                # Kesimpulan Dinamis Grafik CPC
                if mean_cpc > hist_cpc * 1.2:
                    st.error(f"💸 **Membaca Grafik Biaya:** Harga per klik diprediksi **lebih mahal (Merah)**. Biasa terjadi jika *budget* terlalu dipaksakan atau persaingan lelang sedang ketat.")
                elif mean_cpc < hist_cpc * 0.9:
                    st.success(f"🔥 **Membaca Grafik Biaya:** Kabar baik! Harga klik diprediksi **lebih murah (Hijau)**. Algoritma Meta menemukan target audiens yang pas dengan harga diskon.")
                else:
                    st.info("⚖️ **Membaca Grafik Biaya:** Harga klik diprediksi **sangat aman (Biru)** dan tidak ada lonjakan biaya yang mencurigakan.")
            
            with st.expander("🔍 Lihat Grafik Detail CTR (Daya Tarik Iklan)"):
                st.plotly_chart(plot_with_benchmark(forecast_df, "Date", "CTR", "Daya Tarik CTR (vs Historis)", hist_ctr, True), use_container_width=True)
                st.caption("💡 *Insight: Semakin tinggi pilar CTR di atas garis kuning, artinya iklan Anda semakin relevan dan memancing rasa penasaran audiens.*")
            
            st.subheader("Data Detail")
            display_df = forecast_df.copy()
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            display_df["CTR"] = display_df["CTR"].round(4)
            display_df["CPC"] = display_df["CPC"].round(0).astype(int).apply(lambda x: f"Rp {x:,}")
            display_df["Clicks"] = display_df["Clicks"].round(0).astype(int)
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
        else:
            st.subheader("📊 Perbandingan Skenario Budget")
            scenario_adjustments = {name: SCENARIO_BUDGETS[name] for name in selected_scenarios}
            
            with st.spinner("Sedang mensimulasikan berbagai kemungkinan..."):
                scenarios = run_scenario_simulations(budget, funnel, start_date, horizon, metadata, scenario_adjustments, eda_df)
                
            all_forecasts = pd.concat(scenarios.values(), ignore_index=True)
            
            st.subheader("📈 Ringkasan Perbandingan (Estimasi Rentang)")
            summary_data = []
            for scenario_name in selected_scenarios:
                forecast = scenarios[scenario_name]
                s_mean_cpc, s_total_clicks = forecast['CPC'].mean(), forecast['Clicks'].sum()
                s_budget_float = float(forecast.iloc[0]["Budget"].replace("Rp ", "").replace(",", ""))
                
                s_est_spend = min(s_mean_cpc * s_total_clicks, s_budget_float * 1.05)
                if s_est_spend == s_budget_float * 1.05 and s_mean_cpc > 0: s_total_clicks = s_est_spend / s_mean_cpc

                ERR = 0.15
                summary_data.append({
                    "Nama Skenario": scenario_name,
                    "Target Budget": forecast.iloc[0]['Budget'],
                    "Rata-rata Harga Klik (CPC)": f"Rp {s_mean_cpc:,.0f}",
                    "Estimasi Klik Didapat": f"{s_total_clicks*(1-ERR):,.0f} - {s_total_clicks*(1+ERR):,.0f}",
                    "Estimasi Uang Habis": f"Rp {s_est_spend*(1-ERR):,.0f} - Rp {min(s_est_spend*(1+ERR), s_budget_float*1.05):,.0f}"
                })
                
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            st.plotly_chart(px.line(all_forecasts, x="Date", y="Clicks", color="Scenario", markers=True, title="Perbandingan Jumlah Klik Antar Skenario"), use_container_width=True)
            st.plotly_chart(px.line(all_forecasts, x="Date", y="CPC", color="Scenario", markers=True, title="Perbandingan Harga CPC Antar Skenario"), use_container_width=True)

with tab_eda:
    st.header("📊 Data Performa Iklan Masa Lalu (Historis)")
    st.caption("Rata-rata kinerja Meta Ads di masa lalu berdasarkan data ritel elektronik Enterprise.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Data Iklan", len(eda_df))
    with col2: st.metric("Jumlah Campaign", eda_df["ad_set"].nunique())
    with col3: st.metric("Jenis Funnel", eda_df["funnel"].nunique())
    with col4: st.metric("Rentang Waktu", f"{(eda_df['day'].max() - eda_df['day'].min()).days} Hari")

    st.plotly_chart(px.bar(eda_df["funnel"].value_counts().reset_index(), x="funnel", y="count", title="Distribusi Target Iklan (Funnel)"), use_container_width=True)

with tab_info:
    st.header("🧠 Arsitektur Mesin Prediksi (Sistem AI)")
    st.info("""
    **FITUR UTAMA v4.1:**
    1. **Logika Kejenuhan:** Otomatis menyesuaikan harga klik jika budget dipaksa tinggi.
    2. **Smart Absorption:** Menghitung total klik murni secara rasional berdasarkan interaksi iklan (CTR), mengatasi batas maksimal Machine Learning.
    3. **Kalender Dinamis:** Grafik memiliki efek fluktuasi *weekend/weekday* untuk prediksi multi-hari.
    4. **Funnel Benchmark:** Standar indikator Aman/Bahaya otomatis menyesuaikan target kampanye (*Awareness* vs *Conversion*).
    5. **Auto-Insights:** Teks kesimpulan yang pintar menerjemahkan bahasa grafik ke dalam strategi bisnis praktis.
    """)