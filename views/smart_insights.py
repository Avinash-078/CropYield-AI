"""Smart Farming Insights page for CropYield AI."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    CANONICAL_CROPS,
    CANONICAL_STATES,
    STATE_DISTRICT_MAP,
    SOIL_TYPES
)
from utils.model_engine import predict_single, get_model
from utils.visualizer import plot_soil_radar, BASE_LAYOUT, PALETTE


def render_smart_insights():
    st.markdown("""
    <div class="page-banner">
        <h1>💡 Smart Farming Insights & Decision Support</h1>
        <p>Actionable agronomic intelligence: Soil nutrient radar diagnostics, climate change sensitivity simulations, fertilizer efficiency analysis, and harvest revenue forecasting.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Soil N-P-K Radar Health",
        "🌦️ Climate Sensitivity Simulator",
        "💧 Irrigation & Soil Water Dynamics",
        "💰 Farm Economics & Revenue Forecaster"
    ])
    
    with tab1:
        render_soil_health_tab()
        
    with tab2:
        render_climate_simulator_tab()
        
    with tab3:
        render_irrigation_tab()
        
    with tab4:
        render_revenue_tab()


def render_soil_health_tab():
    st.markdown('<div class="section-card"><div class="section-title">🧪 Soil Macro-Nutrient Profile & Balanced Benchmarks</div>', unsafe_allow_html=True)
    
    col_input, col_radar = st.columns([1, 1.2])
    
    with col_input:
        st.write("Tune farm nutrient levels to evaluate deviation from balanced regional agronomic baselines:")
        in_n = st.slider("Nitrogen (N) Content", 20.0, 180.0, 110.0, 5.0, key="soil_n")
        in_p = st.slider("Phosphorus (P) Content", 10.0, 90.0, 45.0, 5.0, key="soil_p")
        in_k = st.slider("Potassium (K) Content", 10.0, 120.0, 75.0, 5.0, key="soil_k")
        
        # Diagnostic Assessment
        n_status = "Optimal" if 90 <= in_n <= 125 else ("Deficient (<90)" if in_n < 90 else "Excessive (>125)")
        p_status = "Optimal" if 40 <= in_p <= 65 else ("Deficient (<40)" if in_p < 40 else "Excessive (>65)")
        k_status = "Optimal" if 55 <= in_k <= 85 else ("Deficient (<55)" if in_k < 55 else "Excessive (>85)")
        
        st.write("")
        st.markdown(f"""
        **Nutrient Diagnostic Summary:**
        - **Nitrogen (N):** `{n_status}` — *Essential for leaf vegetative expansion and chlorophyll synthesis.*
        - **Phosphorus (P):** `{p_status}` — *Promotes deep root development and early seedling vigor.*
        - **Potassium (K):** `{k_status}` — *Regulates stomatal conductance, water retention, and disease resistance.*
        """)
        
    with col_radar:
        st.plotly_chart(plot_soil_radar(in_n, in_p, in_k), use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)


def render_climate_simulator_tab():
    st.markdown('<div class="section-card"><div class="section-title">🌦️ Climate Sensitivity & Extreme Weather Simulator</div>', unsafe_allow_html=True)
    st.write("Simulate how unpredictable weather anomalies (drought, excessive monsoon, or heatwaves) impact crop productivity.")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        sim_crop = st.selectbox("Crop Variety", CANONICAL_CROPS, index=4, key="sim_crop")  # Rice
    with c2:
        sim_state = st.selectbox("State Zone", CANONICAL_STATES, index=4, key="sim_state")  # Punjab
    with c3:
        sim_factor = st.selectbox("Simulate Sensitivity Along:", ["Rainfall (mm)", "Temperature (°C)"])
        
    base_input = {
        "state": sim_state,
        "district": STATE_DISTRICT_MAP.get(sim_state, "Ludhiana"),
        "crop": sim_crop,
        "season": "Kharif",
        "area_hectare": 4.0,
        "rainfall_mm": 1200.0,
        "temperature_c": 28.0,
        "humidity": 70.0,
        "soil_type": "Loamy",
        "nitrogen": 110.0,
        "phosphorus": 50.0,
        "potassium": 65.0,
        "fertilizer_used": 200.0,
        "irrigation": "Yes"
    }
    
    sweep_values = []
    sweep_yields = []
    
    if "Rainfall" in sim_factor:
        sweep_values = np.linspace(200.0, 2400.0, 25)
        for r in sweep_values:
            inp = base_input.copy()
            inp["rainfall_mm"] = r
            sweep_yields.append(predict_single("Random Forest", inp)["predicted_yield"])
        x_label = "Seasonal Rainfall (mm)"
    else:
        sweep_values = np.linspace(16.0, 42.0, 25)
        for t in sweep_values:
            inp = base_input.copy()
            inp["temperature_c"] = t
            sweep_yields.append(predict_single("Random Forest", inp)["predicted_yield"])
        x_label = "Atmospheric Temperature (°C)"
        
    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=sweep_values,
        y=sweep_yields,
        mode="lines+markers",
        line=dict(color="#1b4332", width=3),
        marker=dict(size=6, color="#52b788"),
        name="Simulated Yield Response",
        hovertemplate=f"{x_label}: %{{x:.1f}}<br>Predicted Yield: %{{y:.2f}} t/ha<extra></extra>"
    ))
    
    # Mark maximum optimal point
    max_idx = int(np.argmax(sweep_yields))
    optimal_x = sweep_values[max_idx]
    optimal_y = sweep_yields[max_idx]
    
    fig_sens.add_trace(go.Scatter(
        x=[optimal_x],
        y=[optimal_y],
        mode="markers",
        marker=dict(color="#dc2626", size=12, symbol="star"),
        name=f"Peak Optimum: {optimal_x:.1f} ({optimal_y:.2f} t/ha)"
    ))
    
    fig_sens.update_layout(
        BASE_LAYOUT,
        title=f"<b>Simulated Yield Response Curve for {sim_crop} vs. {sim_factor}</b>",
        xaxis=dict(title=x_label, gridcolor=PALETTE["grid_color"]),
        yaxis=dict(title="Predicted Yield (t/ha)", gridcolor=PALETTE["grid_color"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_sens, use_container_width=True)
    
    st.caption(f"Insight: Model detects optimal agronomic peak for {sim_crop} at approximately **{optimal_x:.1f}** {x_label.split(' ')[-1]}. Deviations below or above show diminishing returns.")
    st.markdown('</div>', unsafe_allow_html=True)


def render_irrigation_tab():
    st.markdown('<div class="section-card"><div class="section-title">💧 Irrigation Impact Across Soil Geologies</div>', unsafe_allow_html=True)
    st.write("Examine the quantitative yield premium generated by supplemental irrigation across diverse soil textures.")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        irrig_crop = st.selectbox("Inspect Crop", CANONICAL_CROPS, index=6, key="irrig_crop")  # Wheat
    with col_sel2:
        irrig_state = st.selectbox("Inspect State", CANONICAL_STATES, index=4, key="irrig_state")  # Punjab
        
    soil_comparison = []
    for s in SOIL_TYPES:
        inp_no = {
            "state": irrig_state, "district": STATE_DISTRICT_MAP.get(irrig_state, "Ludhiana"),
            "crop": irrig_crop, "season": "Rabi", "area_hectare": 5.0, "rainfall_mm": 600.0,
            "temperature_c": 22.0, "humidity": 55.0, "soil_type": s, "nitrogen": 115.0,
            "phosphorus": 55.0, "potassium": 65.0, "fertilizer_used": 200.0, "irrigation": "No"
        }
        inp_yes = inp_no.copy()
        inp_yes["irrigation"] = "Yes"
        
        y_no = predict_single("Random Forest", inp_no)["predicted_yield"]
        y_yes = predict_single("Random Forest", inp_yes)["predicted_yield"]
        gain = y_yes - y_no
        gain_pct = (gain / y_no) * 100 if y_no > 0 else 0.0
        
        soil_comparison.append({
            "Soil Type": s,
            "Rainfed Yield (t/ha)": y_no,
            "Irrigated Yield (t/ha)": y_yes,
            "Irrigation Gain (t/ha)": gain,
            "Yield Lift (%)": gain_pct
        })
        
    df_irrig = pd.DataFrame(soil_comparison)
    
    fig_irrig = go.Figure()
    fig_irrig.add_trace(go.Bar(
        x=df_irrig["Soil Type"],
        y=df_irrig["Rainfed Yield (t/ha)"],
        name="Rainfed (No Irrigation)",
        marker_color="#94a3b8"
    ))
    fig_irrig.add_trace(go.Bar(
        x=df_irrig["Soil Type"],
        y=df_irrig["Irrigated Yield (t/ha)"],
        name="Irrigated Farm",
        marker_color="#2d6a4f"
    ))
    fig_irrig.update_layout(
        BASE_LAYOUT,
        barmode="group",
        title=f"<b>Irrigation Advantage for {irrig_crop} by Soil Classification</b>",
        xaxis=dict(title="Soil Classification"),
        yaxis=dict(title="Yield (t/ha)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_irrig, use_container_width=True)
    
    st.dataframe(df_irrig.round(2), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_revenue_tab():
    st.markdown('<div class="section-card"><div class="section-title">💰 Farm Economics & Revenue Projections</div>', unsafe_allow_html=True)
    st.write("Convert forecasted physical yield into gross agricultural revenue using Indian Minimum Support Price (MSP) & indicative market benchmarks.")
    
    # MSP benchmarks in INR per Quintal (1 Quintal = 0.1 Metric Ton) -> per ton = MSP * 10
    MSP_RATES_PER_TON = {
        "Wheat": 22750,      # ~₹2,275/quintal
        "Rice": 23000,       # ~₹2,300/quintal
        "Cotton": 71210,     # ~₹7,121/quintal
        "Maize": 20900,      # ~₹2,090/quintal
        "Mustard": 56500,    # ~₹5,650/quintal
        "Sugarcane": 3400,   # ~₹340/quintal (FRP)
        "Potato": 14000      # ~₹1,400/quintal
    }
    
    rev_c1, rev_c2, rev_c3 = st.columns(3)
    with rev_c1:
        rev_crop = st.selectbox("Select Crop", CANONICAL_CROPS, index=6, key="rev_crop")  # Wheat
    with rev_c2:
        rev_area = st.number_input("Cultivated Area (Hectares)", 0.5, 50.0, 5.0, 0.5, key="rev_area")
    with rev_c3:
        default_price = MSP_RATES_PER_TON.get(rev_crop, 22000)
        unit_price = st.number_input(f"Estimated Price per Ton (₹)", min_value=1000, max_value=200000, value=default_price, step=500)
        
    sim_input = {
        "state": "Punjab", "district": "Ludhiana", "crop": rev_crop, "season": "Rabi",
        "area_hectare": rev_area, "rainfall_mm": 700.0, "temperature_c": 22.0, "humidity": 60.0,
        "soil_type": "Loamy", "nitrogen": 120.0, "phosphorus": 55.0, "potassium": 70.0,
        "fertilizer_used": 220.0, "irrigation": "Yes"
    }
    
    rev_pred = predict_single("Random Forest", sim_input)
    est_yield = rev_pred["predicted_yield"]
    total_tons = est_yield * rev_area
    gross_revenue = total_tons * unit_price
    
    # Cost estimation (~40% of gross revenue for inputs, labor, seed, fuel)
    estimated_cost = gross_revenue * 0.38
    net_profit = gross_revenue - estimated_cost
    
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Expected Yield", f"{est_yield:.2f} t/ha", delta="Random Forest")
    with r2:
        st.metric("Total Harvest", f"{total_tons:.1f} Metric Tons", delta=f"{rev_area} ha")
    with r3:
        st.metric("Gross Farm Revenue", f"₹{gross_revenue:,.0f}", delta=f"₹{unit_price:,.0f} / Ton")
    with r4:
        st.metric("Estimated Net Margin", f"₹{net_profit:,.0f}", delta="~62% Operating Margin")
        
    st.caption("Disclaimer: Revenue calculations are indicative estimations based on official MSP/market references and do not account for post-harvest transport or regional mandi price volatility.")
    st.markdown('</div>', unsafe_allow_html=True)
