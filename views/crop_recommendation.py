"""Crop Recommendation page for CropYield AI."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    CANONICAL_STATES,
    STATE_DISTRICT_MAP,
    SEASONS,
    SOIL_TYPES,
    IRRIGATION_OPTIONS,
    PRESETS
)
from utils.model_engine import recommend_crops
from utils.visualizer import BASE_LAYOUT, PALETTE


def render_crop_recommendation():
    st.markdown("""
    <div class="page-banner">
        <h1>🌾 Smart Crop Recommendation Engine</h1>
        <p>Harness our champion Random Forest machine learning pipeline to analyze your farm's unique soil chemistry, seasonal window, and meteorological conditions, ranking the most productive crops for cultivation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_input, col_results = st.columns([1, 1.35])
    
    with col_input:
        st.markdown('<div class="section-card"><div class="section-title">🌱 Farm Parameters & Soil Health</div>', unsafe_allow_html=True)
        
        c_state = st.selectbox("Cultivation State", CANONICAL_STATES, index=4)  # Punjab
        c_district = STATE_DISTRICT_MAP.get(c_state, "Ludhiana")
        
        c_season = st.selectbox("Target Cropping Season", SEASONS, index=1)  # Rabi
        c_soil = st.selectbox("Soil Type", SOIL_TYPES, index=2)  # Loamy
        c_irrig = st.selectbox("Irrigation Access", IRRIGATION_OPTIONS, index=0)  # Yes
        
        st.write("---")
        st.markdown("**🌤️ Agro-Climatic Conditions**")
        c_rain = st.slider("Anticipated Rainfall (mm)", 150.0, 2400.0, 750.0, 25.0)
        c_temp = st.slider("Mean Temperature (°C)", 15.0, 42.0, 24.0, 0.5)
        c_humidity = st.slider("Mean Relative Humidity (%)", 30.0, 95.0, 65.0, 1.0)
        
        st.write("---")
        st.markdown("**🧪 Soil Nutrients (N-P-K Index)**")
        c_n = st.slider("Soil Nitrogen (N)", 20.0, 180.0, 115.0, 5.0)
        c_p = st.slider("Soil Phosphorus (P)", 10.0, 90.0, 55.0, 5.0)
        c_k = st.slider("Soil Potassium (K)", 10.0, 120.0, 70.0, 5.0)
        c_fert = st.slider("Planned Fertilizer (kg)", 40.0, 350.0, 220.0, 10.0)
        c_area = st.number_input("Target Farm Acreage (Hectares)", 0.5, 50.0, 4.0, 0.5)
        
        rec_btn = st.button("🔍 Recommend Optimal Crops", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_results:
        # Build environmental payload
        env_payload = {
            "state": c_state,
            "district": c_district,
            "season": c_season,
            "soil_type": c_soil,
            "irrigation": c_irrig,
            "rainfall_mm": c_rain,
            "temperature_c": c_temp,
            "humidity": c_humidity,
            "nitrogen": c_n,
            "phosphorus": c_p,
            "potassium": c_k,
            "fertilizer_used": c_fert,
            "area_hectare": c_area
        }
        
        rec_df = recommend_crops(env_payload)
        
        st.markdown('<div class="section-card"><div class="section-title">🏆 Top Recommended Crops by Predicted Yield</div>', unsafe_allow_html=True)
        
        # Display Top 3 Cards
        top_crop_1 = rec_df.iloc[0]
        top_crop_2 = rec_df.iloc[1]
        top_crop_3 = rec_df.iloc[2]
        
        card_cols = st.columns(3)
        with card_cols[0]:
            st.markdown(f"""
            <div class="crop-card" style="border-left: 4px solid #16a34a; background: #f0fdf4;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="crop-rank" style="background:#16a34a;">#1</span>
                    <span style="font-weight:700; color:#15803d;">Top Pick</span>
                </div>
                <div style="font-size:1.3rem; font-weight:800; color:#081c15; margin:8px 0 2px 0;">{top_crop_1['Crop']}</div>
                <div style="font-size:1.15rem; font-weight:700; color:#16a34a;">{top_crop_1['Expected Yield (t/ha)']:.2f} t/ha</div>
                <div style="font-size:0.75rem; color:#475569; margin-top:4px;">Suitability: <b>{top_crop_1['Suitability Score (%)']:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with card_cols[1]:
            st.markdown(f"""
            <div class="crop-card" style="border-left: 4px solid #2563eb; background: #eff6ff;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="crop-rank" style="background:#2563eb;">#2</span>
                    <span style="font-weight:700; color:#1d4ed8;">Alternative</span>
                </div>
                <div style="font-size:1.3rem; font-weight:800; color:#081c15; margin:8px 0 2px 0;">{top_crop_2['Crop']}</div>
                <div style="font-size:1.15rem; font-weight:700; color:#2563eb;">{top_crop_2['Expected Yield (t/ha)']:.2f} t/ha</div>
                <div style="font-size:0.75rem; color:#475569; margin-top:4px;">Suitability: <b>{top_crop_2['Suitability Score (%)']:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with card_cols[2]:
            st.markdown(f"""
            <div class="crop-card" style="border-left: 4px solid #d97706; background: #fffbeb;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span class="crop-rank" style="background:#d97706;">#3</span>
                    <span style="font-weight:700; color:#b45309;">Viable</span>
                </div>
                <div style="font-size:1.3rem; font-weight:800; color:#081c15; margin:8px 0 2px 0;">{top_crop_3['Crop']}</div>
                <div style="font-size:1.15rem; font-weight:700; color:#d97706;">{top_crop_3['Expected Yield (t/ha)']:.2f} t/ha</div>
                <div style="font-size:0.75rem; color:#475569; margin-top:4px;">Suitability: <b>{top_crop_3['Suitability Score (%)']:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Interactive Plotly Bar Chart
        fig_rec = px.bar(
            rec_df,
            x="Crop",
            y="Expected Yield (t/ha)",
            color="Expected Yield (t/ha)",
            text="Expected Yield (t/ha)",
            color_continuous_scale=[[0, "#74c69d"], [0.5, "#2d6a4f"], [1, "#081c15"]],
            title="<b>Predicted Yield Potential Across All Evaluated Crops</b>"
        )
        fig_rec.update_traces(texttemplate="%{text:.2f} t/ha", textposition="outside")
        fig_rec.update_layout(BASE_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_rec, use_container_width=True)
        
        st.write("")
        st.markdown("#### 📋 Detailed Agronomic Suitability & Cultivation Advisory")
        
        # Table of all crops
        table_view = rec_df[["Rank", "Crop", "Expected Yield (t/ha)", "Suitability Score (%)", "Water Need", "Key Agronomic Tip"]]
        st.dataframe(table_view, use_container_width=True, hide_index=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
