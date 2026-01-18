import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- BAGIAN 1: BACKEND ENGINE (Logic Kita) ---
class BuildingPerformanceEngine:
    def __init__(self, model_path='Energy_Model-Scaler_GBR.pkl'):
        # Load Model & Scaler
        try:
            artifacts = joblib.load(model_path)
            self.model = artifacts['model']
            self.scaler = artifacts['scaler']
        except FileNotFoundError:
            st.error("File Model tidak ditemukan! Pastikan 'Energy_Model-Scaler_GBR.pkl' ada di folder yang sama.")
            st.stop()
        
        # Konstanta Dataset Ecotect
        self.V_REF = 771.75
        
    def process_building(self, u):
        # 1. Hitung Geometri Real (Phase 2)
        H_real = 3.5 if u['lantai'] == 1 else 7.0
        V_real = (u['A1'] * 3.5) + (u['A2'] * 3.5)
        
        # Surface Area Real
        W_real = (u['P1'] * 3.5) + (u['P2'] * 3.5)
        R_real = max(u['A1'], u['A2'])
        F_real = u['A1']
        S_real = W_real + R_real + F_real
        
        # Hitung RC DNA
        RC_real = (6 * (V_real**(0.6667))) / S_real
        Total_Area = u['A1'] + u['A2']
        
        # 2. Miniaturisasi & Prediksi (Phase 3)
        note = ""
        
        # --- JALUR A: NORMAL SHAPE (RC >= 0.62) ---
        if RC_real >= 0.62:
            X1 = min(0.98, RC_real) # Clamping atas
            X2 = (6 * (self.V_REF**(0.6667))) / X1 # Surface Mini
            X5 = H_real
            X4 = 220.5 if X5 == 3.5 else 110.25 # Roof Mini
            X3 = X2 - (2 * X4) # Wall Mini
            
            # Predict
            feat = np.array([[X1, X2, X3, X4, X5, u['O'], u['RG'], u['Dist']]])
            raw = self.model.predict(self.scaler.transform(feat))[0]
            
            # EUI Bridge
            eui_cool = raw[1] / 220.5
            eui_heat = raw[0] / 220.5
            note = "Metode: Interpolasi GBR (Normal Shape)"

        # --- JALUR B: EXTREME SHAPE (RC < 0.62) ---
        else:
            # Base Bungalow Reference (RC 0.62)
            X1_ref = 0.62
            X2_ref = (6 * (self.V_REF**(0.6667))) / 0.62
            X5_ref = 3.5
            X4_ref = 220.5
            X3_ref = X2_ref - (2 * 220.5)
            
            # Predict Base
            feat_ref = np.array([[X1_ref, X2_ref, X3_ref, X4_ref, X5_ref, u['O'], u['RG'], u['Dist']]])
            base_pred = self.model.predict(self.scaler.transform(feat_ref))[0]
            
            eui_cool = base_pred[1] / 220.5
            eui_heat = base_pred[0] / 220.5
            
            # Penalty Factor (Makin gepeng makin boros)
            shape_penalty = 0.62 / RC_real
            orient_penalty = 1.05 if u['O'] in [3, 5] else 1.0
            
            eui_cool *= (shape_penalty * orient_penalty)
            eui_heat *= (shape_penalty * orient_penalty)
            note = f"Metode: Physics Penalty (Extreme Shape | Factor {shape_penalty:.2f}x)"

        # 3. Final Output (Phase 4)
        return {
            'RC': RC_real,
            'Cool_Load': eui_cool * Total_Area,
            'Heat_Load': eui_heat * Total_Area,
            'EUI': eui_cool, # Kita fokus EUI Cooling buat skor
            'Note': note
        }

# --- BAGIAN 2: FRONTEND STREAMLIT (Tampilan) ---

# A. Konfigurasi Halaman
st.set_page_config(page_title="Athena Energy Simulator", page_icon="🏛️", layout="wide")

st.title("🏛️ Athena Energy Simulator")
st.markdown("Simulasi beban energi bangunan di iklim Mediterania (Athena) menggunakan **GBR Physics Engine**.")
st.markdown("---")

# B. Sidebar Input (Phase 1)
with st.sidebar:
    st.header("1. Konfigurasi Bangunan")
    lantai_opt = st.radio("Jumlah Lantai:", [1, 2], horizontal=True)
    
    st.subheader("Dimensi Lantai 1")
    a1 = st.number_input("Luas Area (m2)", value=100.0, step=1.0, key="a1")
    p1 = st.number_input("Keliling (m)", value=40.0, step=0.5, key="p1")
    
    a2, p2 = 0.0, 0.0
    if lantai_opt == 2:
        st.subheader("Dimensi Lantai 2")
        a2 = st.number_input("Luas Area Lt.2 (m2)", value=50.0, step=1.0, key="a2")
        p2 = st.number_input("Keliling Lt.2 (m)", value=30.0, step=0.5, key="p2")
    
    st.header("2. Desain Fasad")
    orientasi = st.selectbox("Orientasi Utama", 
                             options=[2, 3, 4, 5], 
                             format_func=lambda x: {2:"Utara (Cool)", 3:"Timur", 4:"Selatan", 5:"Barat (Hot)"}[x])
    
    glass_ratio = st.slider("Rasio Kaca Total (WWR)", 0.0, 0.40, 0.10, 0.01)
    
    dist_map = {"Merata": 1, "Dominan Utara": 2, "Dominan Timur": 3, "Dominan Selatan": 4, "Dominan Barat": 5}
    glass_dist = st.selectbox("Sebaran Kaca", options=list(dist_map.keys()))

    btn_hitung = st.button("🚀 Hitung Simulasi", type="primary")

# C. Main Area (Phase 4 Output)
if btn_hitung:
    # 1. Collect Input
    user_data = {
        'lantai': lantai_opt,
        'A1': a1, 'P1': p1, 'A2': a2, 'P2': p2,
        'O': orientasi, 'RG': glass_ratio, 'Dist': dist_map[glass_dist]
    }
    
    # 2. Panggil Engine
    engine = BuildingPerformanceEngine() # Load model
    result = engine.process_building(user_data)
    
    # 3. Tampilkan Hasil
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Estimasi Cooling Load", f"{result['Cool_Load']:.2f} kWh")
    with col2:
        st.metric("Estimasi Heating Load", f"{result['Heat_Load']:.2f} kWh")
    with col3:
        # Color Coding EUI
        eui_val = result['EUI']
        if eui_val < 0.18: color = "🟢 Sangat Efisien"
        elif eui_val < 0.28: color = "🟡 Standar"
        else: color = "🔴 Boros"
        st.metric("Efisiensi (EUI Cooling)", f"{eui_val:.2f} kWh/m²", delta=color, delta_color="off")

    # Detail Info
    st.info(f"**Analisis Geometri:** RC Asli = {result['RC']:.3f} | {result['Note']}")
    
    # Visualisasi Sederhana (Bar Chart)
    chart_data = pd.DataFrame({
        'Load Type': ['Cooling', 'Heating'],
        'kWh': [result['Cool_Load'], result['Heat_Load']]
    })
    st.bar_chart(chart_data, x='Load Type', y='kWh', color='#FF4B4B')

else:
    st.info("👈 Masukkan parameter bangunan di sidebar dan tekan tombol Hitung.")