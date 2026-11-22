"""Overview Dashboard page for CropYield AI."""

import streamlit as st
import pandas as pd

from utils.data_loader import (
    load_clean_dataset,
    load_model_performance,
    get_dataset_summary,
    CANONICAL_CROPS,
    CANONICAL_STATES
)
from utils.visualizer import (
    render_kpi_card,
    render_kpi_grid,
    plot_crop_yield_bar,
    plot_state_yield_comparison
)
from utils.model_engine import predict_single


def render_overview():
    # Page Banner
    st.markdown("""
    <div class="page-banner">
        <h1>🌾 CropYield AI — Agricultural Intelligence Dashboard</h1>
        <p>Enterprise-grade predictive platform leveraging multi-model machine learning for precision crop yield forecasting, soil health optimization, and agronomic risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Summary Metrics
    summary = get_dataset_summary()
    
    st.markdown(render_kpi_grid(
        render_kpi_card(
            title="Total Observations",
            value=f"{summary['total_records']:,}",
            badge_text="Verified Data",
            badge_type="green",
            subtitle="Cleaned field trials"
        ),
        render_kpi_card(
            title="Champion Model",
            value=f"{summary['best_model_name']}",
            badge_text=f"R² {summary['best_model_r2']:.3f}",
            badge_type="green",
            subtitle=f"RMSE: {summary['best_model_rmse']:.3f} t/ha"
        ),
        render_kpi_card(
            title="Mean Dataset Yield",
            value=f"{summary['mean_yield']:.2f} t/ha",
            badge_text="National Avg",
            badge_type="blue",
            subtitle=f"Range: {summary['min_yield']:.1f} – {summary['max_yield']:.1f} t/ha"
        ),
        render_kpi_card(
            title="Agronomic Coverage",
            value=f"{summary['total_crops']} Crops",
            badge_text=f"{summary['total_states']} States",
            badge_type="purple",
            subtitle="Diverse agro-climatic zones"
        ),
    ), unsafe_allow_html=True)
        
    st.write("")
    
    # Charts Row — equal width for balanced alignment
    df = load_clean_dataset()
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        st.plotly_chart(plot_crop_yield_bar(df), use_container_width=True)
        
    with col_chart2:
        st.plotly_chart(plot_state_yield_comparison(df), use_container_width=True)
        
    st.write("")
    
    # Quick Assessment Interactive Card
    st.markdown('<div class="section-card"><div class="section-title">⚡ Instant Farm Yield Estimator (Live Inference)</div>', unsafe_allow_html=True)
    
    q_col1, q_col2, q_col3, q_col4 = st.columns(4)
    with q_col1:
        quick_crop = st.selectbox("Select Crop", CANONICAL_CROPS, index=4)  # Rice
    with q_col2:
        quick_state = st.selectbox("Select State", CANONICAL_STATES, index=4)  # Punjab
    with q_col3:
        quick_area = st.number_input("Cultivated Area (Hectares)", min_value=0.5, max_value=50.0, value=4.0, step=0.5)
    with q_col4:
        quick_irrigation = st.selectbox("Irrigation Facility", ["Yes", "No"], index=0)
        
    quick_sample = {
        "state": quick_state,
        "district": "Ludhiana" if quick_state == "Punjab" else "Patna",
        "crop": quick_crop,
        "season": "Kharif" if quick_crop in ["Rice", "Cotton"] else "Rabi",
        "area_hectare": quick_area,
        "rainfall_mm": 1250.0,
        "temperature_c": 28.0,
        "humidity": 70.0,
        "soil_type": "Loamy",
        "nitrogen": 110.0,
        "phosphorus": 55.0,
        "potassium": 65.0,
        "fertilizer_used": 210.0,
        "irrigation": quick_irrigation
    }
    
    res = predict_single("Random Forest", quick_sample)
    
    r_col1, r_col2, r_col3 = st.columns([1, 1, 1.2])
    with r_col1:
        st.metric("Estimated Yield", f"{res['predicted_yield']:.2f} t/ha", delta=f"{res['yield_category']}")
    with r_col2:
        st.metric("Total Expected Harvest", f"{res['total_production']:.2f} Metric Tons", delta=f"{quick_area} Hectares")
    with r_col3:
        st.metric("Model Confidence (R²)", f"{res['r2_score'] * 100:.1f}%", delta=f"Margin ±{res['rmse']:.2f} t/ha")
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Workflow and Architecture Overview
    st.markdown('<div class="section-card"><div class="section-title">🏗️ End-to-End System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-grid">
        <div class="arch-card">
            <div class="arch-card-title">1. Agronomic Ingestion</div>
            <p class="arch-card-desc">14 biological, pedological, and meteorological attributes capturing real farm dynamics across 6 Indian states.</p>
        </div>
        <div class="arch-card">
            <div class="arch-card-title">2. Pipeline Preprocessing</div>
            <p class="arch-card-desc">Median/Modal imputation, StandardScaler normalization, and OneHotEncoder handling unknown categories robustly.</p>
        </div>
        <div class="arch-card">
            <div class="arch-card-title">3. Multi-Model Inference</div>
            <p class="arch-card-desc">Simultaneous scoring via Random Forest (Champion), Decision Trees, Support Vector Regression, and Linear Regression.</p>
        </div>
        <div class="arch-card">
            <div class="arch-card-title">4. Actionable Advisory</div>
            <p class="arch-card-desc">Crop suitability ranking, nutrient balance guidance, climate sensitivity simulation, and harvest tonnage projection.</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
