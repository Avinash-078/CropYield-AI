"""Prediction History page for CropYield AI."""

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.history_manager import (
    get_history,
    clear_history,
    get_history_stats
)
from utils.visualizer import render_kpi_card, render_kpi_grid, BASE_LAYOUT, PALETTE


def render_prediction_history():
    st.markdown("""
    <div class="page-banner">
        <h1>📜 Inference Log & Prediction History</h1>
        <p>Comprehensive audit record of all crop yield predictions generated during the current and prior analytical sessions. Tracks input parameters, applied models, and tonnage outcomes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    stats = get_history_stats()
    
    # History Summary Cards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(render_kpi_card(
            title="Total Predictions Logged",
            value=str(stats["total_runs"]),
            badge_text="Real-time log",
            badge_type="green",
            subtitle="Session & disk records"
        ), unsafe_allow_html=True)
    with col_k2:
        st.markdown(render_kpi_card(
            title="Average Predicted Yield",
            value=f"{stats['avg_predicted_yield']:.2f} t/ha" if stats["total_runs"] > 0 else "0.00 t/ha",
            badge_text="Mean output",
            badge_type="blue",
            subtitle="Across all logged queries"
        ), unsafe_allow_html=True)
    with col_k3:
        st.markdown(render_kpi_card(
            title="Total Harvest Projected",
            value=f"{stats['total_tonnage']:,.1f} T",
            badge_text="Metric Tons",
            badge_type="purple",
            subtitle="Cumulative production"
        ), unsafe_allow_html=True)
    with col_k4:
        st.markdown(render_kpi_card(
            title="Dominant Crop Query",
            value=str(stats["top_crop"]),
            badge_text=f"Model: {stats['top_model']}",
            badge_type="amber",
            subtitle="Most evaluated variety"
        ), unsafe_allow_html=True)
        
    st.write("")
    
    history_df = get_history()
    
    if history_df.empty:
        st.info("No prediction history recorded yet. Head over to the **Crop Yield Prediction** page to run your first forecast!")
        return
        
    st.markdown('<div class="section-card"><div class="section-title">🔍 Historical Queries & Filtering</div>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        all_crops = sorted(history_df["Crop"].dropna().unique().tolist())
        sel_crops = st.multiselect("Filter by Crop:", all_crops, default=[])
    with col_f2:
        all_models = sorted(history_df["Model"].dropna().unique().tolist())
        sel_models = st.multiselect("Filter by Model:", all_models, default=[])
    with col_f3:
        all_cats = sorted(history_df["Yield Category"].dropna().unique().tolist())
        sel_cats = st.multiselect("Filter by Category:", all_cats, default=[])
        
    filtered_history = history_df.copy()
    if sel_crops:
        filtered_history = filtered_history[filtered_history["Crop"].isin(sel_crops)]
    if sel_models:
        filtered_history = filtered_history[filtered_history["Model"].isin(sel_models)]
    if sel_cats:
        filtered_history = filtered_history[filtered_history["Yield Category"].isin(sel_cats)]
        
    st.dataframe(filtered_history, use_container_width=True, height=350)
    
    # Action buttons
    col_btn1, col_btn2, _ = st.columns([1, 1, 2])
    with col_btn1:
        csv_data = filtered_history.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Export History as CSV",
            data=csv_data,
            file_name="crop_yield_prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_btn2:
        if st.button("🗑️ Clear Prediction History", use_container_width=True):
            clear_history()
            st.success("Prediction history cleared successfully.")
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Historical Trend Chart
    if len(history_df) >= 2:
        st.markdown('<div class="section-card"><div class="section-title">📈 Query Productivity Distribution</div>', unsafe_allow_html=True)
        fig_trend = px.scatter(
            history_df,
            x="Timestamp",
            y="Predicted Yield (t/ha)",
            color="Crop",
            size="Total Production (Tons)",
            hover_data=["State", "Soil Type", "Model"],
            title="<b>Timeline of Predicted Crop Yields</b>",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_trend.update_layout(BASE_LAYOUT)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
