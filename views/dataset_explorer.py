"""Dataset Explorer page for CropYield AI."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import (
    load_clean_dataset,
    load_messy_dataset,
    DATA_DICTIONARY,
    CANONICAL_CROPS,
    CANONICAL_STATES
)
from utils.visualizer import plot_correlation_heatmap, render_kpi_card, render_kpi_grid, BASE_LAYOUT, PALETTE


def render_dataset_explorer():
    st.markdown("""
    <div class="page-banner">
        <h1>🔍 Dataset Explorer — Field Data Repository</h1>
        <p>Explore, filter, and inspect the foundational 500-sample agricultural dataset. Compare the raw messy collection against the pipeline-cleaned production data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dataset Mode Selector
    ds_mode = st.radio(
        "Select Dataset View:",
        ["Cleaned Dataset (crop_yield_cleaned.xlsx)", "Raw Messy Dataset (Crop_Yield_Messy_Dataset_500.xlsx)"],
        horizontal=True
    )
    
    clean_df = load_clean_dataset()
    messy_df = load_messy_dataset()
    
    # Show Data Cleaning Pipeline Stats
    duplicates_count = clean_df.duplicated().sum() if "Cleaned" in ds_mode else messy_df.duplicated().sum()
    missing_count = clean_df.isnull().sum().sum() if "Cleaned" in ds_mode else messy_df.isnull().sum().sum()
    cols_count = len(clean_df.columns) if "Cleaned" in ds_mode else len(messy_df.columns)

    st.markdown(render_kpi_grid(
        render_kpi_card(
            title="Total Observations",
            value=f"{len(clean_df)}" if "Cleaned" in ds_mode else f"{len(messy_df)}",
            badge_text="500 Field Trials",
            badge_type="green",
            subtitle="Rows captured"
        ),
        render_kpi_card(
            title="Duplicate Records",
            value=str(duplicates_count),
            badge_text="Deduplicated" if "Cleaned" in ds_mode else "10 Duplicates",
            badge_type="green" if duplicates_count == 0 else "amber",
            subtitle="Zero duplicates in cleaned"
        ),
        render_kpi_card(
            title="Missing Cell Values",
            value=str(missing_count),
            badge_text="Imputed" if "Cleaned" in ds_mode else "25 Nulls in Raw",
            badge_type="blue" if "Cleaned" in ds_mode else "amber",
            subtitle="Pipeline handled"
        ),
        render_kpi_card(
            title="Total Attributes",
            value=f"{cols_count} Columns",
            badge_text="14 Feats + 1 Target",
            badge_type="purple",
            subtitle="Normalized snake_case"
        ),
    ), unsafe_allow_html=True)
        
    st.write("")
    
    active_df = clean_df if "Cleaned" in ds_mode else messy_df
    
    # Filtering Section
    st.markdown('<div class="section-card"><div class="section-title">⚡ Interactive Dataset Filtering</div>', unsafe_allow_html=True)
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        state_col = "state" if "state" in active_df.columns else "State"
        avail_states = sorted(active_df[state_col].dropna().unique().tolist())
        selected_states = st.multiselect("Filter by State:", avail_states, default=[])
    with col_f2:
        crop_col = "crop" if "crop" in active_df.columns else "Crop"
        avail_crops = sorted(active_df[crop_col].dropna().unique().tolist())
        selected_crops = st.multiselect("Filter by Crop:", avail_crops, default=[])
    with col_f3:
        season_col = "season" if "season" in active_df.columns else "Season"
        avail_seasons = sorted(active_df[season_col].dropna().unique().tolist())
        selected_seasons = st.multiselect("Filter by Season:", avail_seasons, default=[])
    with col_f4:
        irrig_col = "irrigation" if "irrigation" in active_df.columns else "Irrigation"
        avail_irrig = sorted(active_df[irrig_col].dropna().unique().tolist())
        selected_irrig = st.multiselect("Filter by Irrigation:", avail_irrig, default=[])
        
    filtered_df = active_df.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df[state_col].isin(selected_states)]
    if selected_crops:
        filtered_df = filtered_df[filtered_df[crop_col].isin(selected_crops)]
    if selected_seasons:
        filtered_df = filtered_df[filtered_df[season_col].isin(selected_seasons)]
    if selected_irrig:
        filtered_df = filtered_df[filtered_df[irrig_col].isin(selected_irrig)]
        
    st.caption(f"Showing **{len(filtered_df)}** matching rows out of {len(active_df)} total.")
    st.dataframe(filtered_df, use_container_width=True, height=320)
    
    # Download filtered button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Current Filtered Table as CSV",
        data=csv_data,
        file_name="crop_yield_filtered_data.csv",
        mime="text/csv"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Statistical Profiling Tabs
    st.markdown('<div class="section-card"><div class="section-title">📊 Statistical Profiling & Agronomic Distributions</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Descriptive Statistics",
        "🌊 Distribution Analysis",
        "📦 Outlier Boxplots",
        "🔥 Correlation Matrix",
        "📖 Data Dictionary"
    ])
    
    with tab1:
        st.write("Summary statistics for numerical features across the dataset:")
        num_stats = clean_df.select_dtypes(include=np.number).describe().T
        num_stats["IQR"] = num_stats["75%"] - num_stats["25%"]
        st.dataframe(num_stats.round(2), use_container_width=True)
        
    with tab2:
        dist_feature = st.selectbox(
            "Select Numerical Attribute to Inspect Distribution:",
            ["rainfall_mm", "temperature_c", "humidity", "nitrogen", "phosphorus", "potassium", "fertilizer_used", "yield_ton_per_hectare"]
        )
        fig_dist = px.histogram(
            clean_df,
            x=dist_feature,
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#2d6a4f"],
            title=f"Distribution & Kernel Density of {dist_feature}"
        )
        fig_dist.update_layout(BASE_LAYOUT)
        st.plotly_chart(fig_dist, use_container_width=True)
        
    with tab3:
        st.write("Boxplot analysis for detecting extreme values in field records:")
        box_cols = ["rainfall_mm", "nitrogen", "potassium", "fertilizer_used", "yield_ton_per_hectare"]
        fig_box = go.Figure()
        for col in box_cols:
            fig_box.add_trace(go.Box(
                y=clean_df[col],
                name=col,
                marker_color="#2d6a4f",
                boxmean="sd"
            ))
        fig_box.update_layout(
            BASE_LAYOUT,
            title="<b>Outlier Inspection Across Agricultural Attributes</b>",
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with tab4:
        st.plotly_chart(plot_correlation_heatmap(clean_df), use_container_width=True)
        
    with tab5:
        dict_rows = [{"Column Name": col, "Description": desc} for col, desc in DATA_DICTIONARY.items()]
        st.dataframe(pd.DataFrame(dict_rows), use_container_width=True, height=350)
        
    st.markdown('</div>', unsafe_allow_html=True)
