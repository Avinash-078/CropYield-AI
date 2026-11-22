"""Crop Yield Prediction page for CropYield AI."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    CANONICAL_CROPS,
    CANONICAL_STATES,
    STATE_DISTRICT_MAP,
    SEASONS,
    SOIL_TYPES,
    IRRIGATION_OPTIONS,
    PRESETS,
    load_feature_columns
)
from utils.model_engine import (
    predict_single,
    predict_all_models,
    predict_batch
)
from utils.history_manager import add_prediction
from utils.visualizer import BASE_LAYOUT, PALETTE


def render_yield_prediction():
    st.markdown("""
    <div class="page-banner">
        <h1>🌱 Crop Yield Prediction — Precision Inference Engine</h1>
        <p>Input farm soil characteristics, environmental weather conditions, and crop management practices to generate real-time yield forecasts using our verified machine learning pipelines.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_single, tab_batch = st.tabs([
        "🚜 Single Farm Estimation",
        "📂 Batch Bulk Prediction (CSV/Excel)"
    ])
    
    with tab_single:
        render_single_farm_prediction()
        
    with tab_batch:
        render_batch_prediction()


def render_single_farm_prediction():
    # Preset Selector Row
    st.markdown("#### ⚡ Quick-Load Regional Presets")
    preset_cols = st.columns(4)
    preset_names = list(PRESETS.keys())
    
    # Store form state in session
    if "farm_input" not in st.session_state:
        st.session_state.farm_input = PRESETS[preset_names[0]].copy()
        
    for i, name in enumerate(preset_names):
        with preset_cols[i]:
            if st.button(f"📌 {name.split(' (')[0]}", key=f"preset_btn_{i}", use_container_width=True):
                st.session_state.farm_input = PRESETS[name].copy()
                st.rerun()
                
    st.write("")
    
    cur = st.session_state.farm_input
    
    with st.form("yield_prediction_form"):
        st.markdown('<div class="section-title">📍 1. Geographic & Soil Classification</div>', unsafe_allow_html=True)
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            state_idx = CANONICAL_STATES.index(cur["state"]) if cur["state"] in CANONICAL_STATES else 0
            state = st.selectbox("State of Cultivation", CANONICAL_STATES, index=state_idx)
        with col_g2:
            default_dist = STATE_DISTRICT_MAP.get(state, "Pune")
            district = st.text_input("District / Administrative Zone", value=default_dist)
        with col_g3:
            soil_idx = SOIL_TYPES.index(cur["soil_type"]) if cur["soil_type"] in SOIL_TYPES else 2
            soil_type = st.selectbox("Pedological Soil Type", SOIL_TYPES, index=soil_idx)
            
        st.markdown('<div class="section-title" style="margin-top: 1rem;">🌾 2. Crop Selection & Land Area</div>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            crop_idx = CANONICAL_CROPS.index(cur["crop"]) if cur["crop"] in CANONICAL_CROPS else 0
            crop = st.selectbox("Target Agricultural Crop", CANONICAL_CROPS, index=crop_idx)
        with col_c2:
            season_idx = SEASONS.index(cur["season"]) if cur["season"] in SEASONS else 0
            season = st.selectbox("Cropping Season", SEASONS, index=season_idx)
        with col_c3:
            area_hectare = st.number_input(
                "Cultivated Land Area (Hectares)",
                min_value=0.5,
                max_value=100.0,
                value=float(cur["area_hectare"]),
                step=0.5
            )
        with col_c4:
            irrig_idx = IRRIGATION_OPTIONS.index(cur["irrigation"]) if cur["irrigation"] in IRRIGATION_OPTIONS else 0
            irrigation = st.selectbox("Irrigation Facility Available?", IRRIGATION_OPTIONS, index=irrig_idx)
            
        st.markdown('<div class="section-title" style="margin-top: 1rem;">🌤️ 3. Weather & Agro-Climatic Conditions</div>', unsafe_allow_html=True)
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            rainfall_mm = st.slider(
                "Seasonal Cumulative Rainfall (mm)",
                min_value=100.0,
                max_value=2500.0,
                value=float(cur["rainfall_mm"]),
                step=25.0
            )
        with col_w2:
            temperature_c = st.slider(
                "Mean Temperature (°C)",
                min_value=15.0,
                max_value=45.0,
                value=float(cur["temperature_c"]),
                step=0.5
            )
        with col_w3:
            humidity = st.slider(
                "Average Relative Humidity (%)",
                min_value=30.0,
                max_value=100.0,
                value=float(cur["humidity"]),
                step=1.0
            )
            
        st.markdown('<div class="section-title" style="margin-top: 1rem;">🧪 4. Soil Nutrient Index & Fertilizer Management</div>', unsafe_allow_html=True)
        col_n1, col_n2, col_n3, col_n4 = st.columns(4)
        with col_n1:
            nitrogen = st.slider("Nitrogen (N) [kg/ha]", min_value=20.0, max_value=180.0, value=float(cur["nitrogen"]), step=5.0)
        with col_n2:
            phosphorus = st.slider("Phosphorus (P) [kg/ha]", min_value=10.0, max_value=90.0, value=float(cur["phosphorus"]), step=5.0)
        with col_n3:
            potassium = st.slider("Potassium (K) [kg/ha]", min_value=10.0, max_value=120.0, value=float(cur["potassium"]), step=5.0)
        with col_n4:
            fertilizer_used = st.slider("Total Fertilizer Applied [kg]", min_value=40.0, max_value=350.0, value=float(cur["fertilizer_used"]), step=10.0)
            
        st.markdown('<div class="section-title" style="margin-top: 1rem;">⚙️ 5. Machine Learning Model Selection</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            selected_model = st.selectbox(
                "Choose Primary Regressor Pipeline:",
                ["Random Forest (Recommended Champion)", "Linear Regression", "SVR", "Decision Tree"]
            )
            clean_model_name = selected_model.split(" (")[0]
            
        with col_m2:
            st.write("")
            st.write("")
            submit_btn = st.form_submit_button("🚀 Run Live Crop Yield Prediction", use_container_width=True)
            
    # Process Prediction
    if submit_btn:
        input_payload = {
            "state": state,
            "district": district,
            "crop": crop,
            "season": season,
            "area_hectare": area_hectare,
            "rainfall_mm": rainfall_mm,
            "temperature_c": temperature_c,
            "humidity": humidity,
            "soil_type": soil_type,
            "nitrogen": nitrogen,
            "phosphorus": phosphorus,
            "potassium": potassium,
            "fertilizer_used": fertilizer_used,
            "irrigation": irrigation
        }
        
        # Save current state
        st.session_state.farm_input = input_payload.copy()
        
        # Primary Model Inference
        pred_res = predict_single(clean_model_name, input_payload)
        
        # Log to Persistent History
        add_prediction(input_payload, pred_res)
        
        st.write("")
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        
        # Yield Hero Output Card
        st.markdown(f"""
        <div class="yield-hero-card">
            <h3>Forecasted Crop Productivity ({clean_model_name})</h3>
            <div class="yield-hero-number">{pred_res['predicted_yield']:.2f}</div>
            <div class="yield-hero-unit">Metric Tons per Hectare (t/ha)</div>
            <div style="margin-top: 1rem; font-size: 0.95rem; color: #d8f3dc;">
                Estimated Confidence Band: <b>{pred_res['lower_bound']:.2f} – {pred_res['upper_bound']:.2f} t/ha</b> (±RMSE)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary KPI cards
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric(
                label="Total Estimated Harvest",
                value=f"{pred_res['total_production']:.2f} Tons",
                delta=f"Over {area_hectare:.1f} hectares"
            )
        with k2:
            st.metric(
                label="Yield Classification",
                value=pred_res['yield_category'].split(" (")[0],
                delta=f"{pred_res['yield_category'].split('(')[-1].replace(')', '')}"
            )
        with k3:
            st.metric(
                label="Model Benchmark R²",
                value=f"{pred_res['r2_score'] * 100:.1f}%",
                delta=f"RMSE: {pred_res['rmse']:.3f} t/ha"
            )
            
        st.write("")
        st.markdown("#### 🔄 Cross-Model Comparison for Current Input")
        st.write("Real-time inference generated by all four models on this exact configuration:")
        
        comp_df = predict_all_models(input_payload)
        st.dataframe(comp_df, use_container_width=True)
        
        # Comparative bar chart
        fig_comp = px.bar(
            comp_df,
            x="Model",
            y="Predicted Yield (t/ha)",
            color="Predicted Yield (t/ha)",
            text="Predicted Yield (t/ha)",
            color_continuous_scale=[[0, "#74c69d"], [1, "#1b4332"]],
            title="<b>Predicted Yield Across All 4 Machine Learning Models</b>"
        )
        fig_comp.update_traces(texttemplate="%{text:.2f} t/ha", textposition="outside")
        fig_comp.update_layout(BASE_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_batch_prediction():
    st.markdown('<div class="section-card"><div class="section-title">📂 Batch Agricultural Predictions</div>', unsafe_allow_html=True)
    st.write("Upload a CSV or Excel spreadsheet containing farm records. The system will run real inference using the selected pipeline.")
    
    # Download sample template
    feature_cols = load_feature_columns()
    sample_rows = [
        {
            "state": "Punjab", "district": "Ludhiana", "crop": "Wheat", "season": "Rabi",
            "area_hectare": 5.0, "rainfall_mm": 650.0, "temperature_c": 22.0, "humidity": 60.0,
            "soil_type": "Loamy", "nitrogen": 130.0, "phosphorus": 60.0, "potassium": 75.0,
            "fertilizer_used": 240.0, "irrigation": "Yes"
        },
        {
            "state": "Maharashtra", "district": "Pune", "crop": "Sugarcane", "season": "Rabi",
            "area_hectare": 6.5, "rainfall_mm": 1200.0, "temperature_c": 30.0, "humidity": 72.0,
            "soil_type": "Black", "nitrogen": 150.0, "phosphorus": 70.0, "potassium": 90.0,
            "fertilizer_used": 280.0, "irrigation": "Yes"
        }
    ]
    sample_template_df = pd.DataFrame(sample_rows, columns=feature_cols)
    template_csv = sample_template_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "📥 Download Standard Batch Template (CSV)",
        data=template_csv,
        file_name="crop_yield_batch_template.csv",
        mime="text/csv"
    )
    
    st.write("")
    uploaded_file = st.file_uploader("Upload CSV or Excel Spreadsheet", type=["csv", "xlsx"])
    
    batch_model = st.selectbox(
        "Batch Regressor Pipeline:",
        ["Random Forest", "Linear Regression", "SVR", "Decision Tree"],
        key="batch_model_select"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
                
            st.write(f"Loaded **{len(df_upload)}** records.")
            
            if st.button("⚡ Execute Batch Predictions", key="exec_batch_btn"):
                batch_results = predict_batch(batch_model, df_upload)
                
                st.success(f"Successfully processed {len(batch_results)} predictions with {batch_model}!")
                st.dataframe(batch_results, use_container_width=True)
                
                # Download results
                res_csv = batch_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Prediction Results as CSV",
                    data=res_csv,
                    file_name="crop_yield_batch_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing batch file: {e}")
            
    st.markdown('</div>', unsafe_allow_html=True)
