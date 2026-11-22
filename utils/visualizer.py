"""Visualizer module for CropYield AI dashboard.
Builds premium, interactive Plotly figures using a forest green and white theme.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Theme palette
PALETTE = {
    "deep": "#081c15",
    "forest": "#1b4332",
    "emerald": "#2d6a4f",
    "accent": "#40916c",
    "mint": "#52b788",
    "light": "#74c69d",
    "pale": "#d8f3dc",
    "bg_subtle": "#f7faf8",
    "white": "#ffffff",
    "gray_text": "#475569",
    "grid_color": "#e2e8f0"
}

BASE_LAYOUT = dict(
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#1e293b"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#1b4332",
        font_size=13,
        font_family="Plus Jakarta Sans, sans-serif",
        font_color="#ffffff"
    )
)


def render_kpi_card(title: str, value: str, badge_text: str = "", badge_type: str = "green", subtitle: str = "") -> str:
    """Generate HTML snippet for a premium SaaS KPI card (stacked layout, anti-overflow).
    Badge and subtitle are placed in a flex-column footer that never overlaps the value.
    """
    badge_class = f"badge-{badge_type}"
    badge_html = f'<span class="kpi-badge {badge_class}">{badge_text}</span>' if badge_text else ''
    sub_html = f'<div class="kpi-subtitle">{subtitle}</div>' if subtitle else ''

    return f"""<div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" title="{value}">{value}</div>
        <div class="kpi-footer">{badge_html}{sub_html}</div>
    </div>"""


def render_kpi_grid(*cards_html: str) -> str:
    """Wrap KPI card HTML strings in a responsive CSS-grid container."""
    return f'<div class="kpi-grid">{"".join(cards_html)}</div>'


def plot_crop_yield_bar(df: pd.DataFrame) -> go.Figure:
    """Plot average crop yield across major crops."""
    crop_stats = df.groupby("crop")["yield_ton_per_hectare"].agg(["mean", "std", "count"]).reset_index()
    crop_stats = crop_stats.sort_values("mean", ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=crop_stats["crop"],
        y=crop_stats["mean"],
        text=[f"{v:.2f} t/ha" for v in crop_stats["mean"]],
        textposition="outside",
        marker=dict(
            color=crop_stats["mean"],
            colorscale=[[0, "#52b788"], [0.5, "#2d6a4f"], [1, "#1b4332"]],
            line=dict(color="#081c15", width=1)
        ),
        hovertemplate="<b>%{x}</b><br>Average Yield: %{y:.2f} t/ha<extra></extra>"
    ))
    
    fig.update_layout(
        BASE_LAYOUT,
        title="<b>Average Yield by Agricultural Crop (t/ha)</b>",
        xaxis=dict(title="", tickangle=-15, gridcolor=PALETTE["grid_color"]),
        yaxis=dict(title="Metric Tons / Hectare", gridcolor=PALETTE["grid_color"], range=[0, max(crop_stats["mean"]) * 1.25]),
        showlegend=False
    )
    return fig


def plot_state_yield_comparison(df: pd.DataFrame) -> go.Figure:
    """Plot regional yield comparison across Indian States."""
    state_stats = df.groupby("state")["yield_ton_per_hectare"].mean().reset_index()
    state_stats = state_stats.sort_values("yield_ton_per_hectare", ascending=True)
    
    fig = go.Figure(go.Bar(
        y=state_stats["state"],
        x=state_stats["yield_ton_per_hectare"],
        orientation="h",
        text=[f"{v:.2f} t/ha" for v in state_stats["yield_ton_per_hectare"]],
        textposition="outside",
        marker=dict(
            color=state_stats["yield_ton_per_hectare"],
            colorscale=[[0, "#74c69d"], [1, "#1b4332"]],
            line=dict(color="#1b4332", width=1)
        ),
        hovertemplate="State: <b>%{y}</b><br>Average Yield: %{x:.2f} t/ha<extra></extra>"
    ))
    
    fig.update_layout(
        BASE_LAYOUT,
        title="<b>Regional Agricultural Productivity by State</b>",
        xaxis=dict(title="Mean Yield (t/ha)", gridcolor=PALETTE["grid_color"], range=[0, max(state_stats["yield_ton_per_hectare"]) * 1.25]),
        yaxis=dict(title="", gridcolor=PALETTE["grid_color"]),
        showlegend=False
    )
    return fig


def plot_model_metrics(perf_df: pd.DataFrame) -> go.Figure:
    """Plot side-by-side R² Score vs RMSE for all trained models."""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="R² Score (Higher is Better)",
        x=perf_df["Model"],
        y=perf_df["R2 Score"],
        text=[f"{v:.3f}" for v in perf_df["R2 Score"]],
        textposition="outside",
        marker_color="#2d6a4f"
    ))
    
    fig.add_trace(go.Bar(
        name="RMSE (Lower is Better)",
        x=perf_df["Model"],
        y=perf_df["RMSE"],
        text=[f"{v:.3f}" for v in perf_df["RMSE"]],
        textposition="outside",
        marker_color="#74c69d"
    ))
    
    fig.update_layout(
        BASE_LAYOUT,
        title="<b>Benchmark Performance Comparison Across Models</b>",
        barmode="group",
        xaxis=dict(title="", gridcolor=PALETTE["grid_color"]),
        yaxis=dict(title="Score / Error Magnitude", gridcolor=PALETTE["grid_color"], range=[0, 1.15]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_actual_vs_predicted(test_df: pd.DataFrame, model_name: str) -> go.Figure:
    """Generate Actual vs Predicted scatter plot with 45-degree ideal fit line."""
    pred_col = f"Pred_{model_name.replace(' ', '_')}"
    if pred_col not in test_df.columns:
        pred_col = test_df.columns[test_df.columns.str.startswith("Pred_")][0]
        
    actual = test_df["Actual_Yield"]
    predicted = test_df[pred_col]
    
    min_val = min(actual.min(), predicted.min()) * 0.9
    max_val = max(actual.max(), predicted.max()) * 1.1
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode="markers",
        marker=dict(
            color="#2d6a4f",
            size=9,
            opacity=0.75,
            line=dict(color="#081c15", width=1)
        ),
        name="Test Observations",
        hovertemplate="Actual: %{x:.2f} t/ha<br>Predicted: %{y:.2f} t/ha<extra></extra>"
    ))
    
    # Ideal 1:1 line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color="#dc2626", dash="dash", width=2),
        name="Ideal Perfect Prediction (y = x)"
    ))
    
    fig.update_layout(
        BASE_LAYOUT,
        title=f"<b>Actual vs. Predicted Yield — {model_name}</b>",
        xaxis=dict(title="Actual Crop Yield (t/ha)", range=[min_val, max_val], gridcolor=PALETTE["grid_color"]),
        yaxis=dict(title="Predicted Crop Yield (t/ha)", range=[min_val, max_val], gridcolor=PALETTE["grid_color"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Horizontal bar chart for Random Forest feature importances."""
    top_fi = fi_df.head(top_n).sort_values("Importance", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=top_fi["Importance %"],
        y=top_fi["Feature"],
        orientation="h",
        text=[f"{v:.1f}%" for v in top_fi["Importance %"]],
        textposition="outside",
        marker=dict(
            color=top_fi["Importance %"],
            colorscale=[[0, "#74c69d"], [0.5, "#40916c"], [1, "#1b4332"]],
            line=dict(color="#081c15", width=1)
        ),
        hovertemplate="Feature: <b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>"
    ))
    
    fig.update_layout(
        BASE_LAYOUT,
        title="<b>Random Forest Feature Importance (% Relative Weight)</b>",
        xaxis=dict(title="Relative Importance (%)", gridcolor=PALETTE["grid_color"], range=[0, max(top_fi["Importance %"]) * 1.2]),
        yaxis=dict(title="", gridcolor=PALETTE["grid_color"]),
        showlegend=False
    )
    return fig


def plot_soil_radar(n: float, p: float, k: float) -> go.Figure:
    """Radar chart comparing user N-P-K nutrient levels against balanced benchmarks."""
    categories = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]
    
    # Benchmarks: N ~ 100, P ~ 50, K ~ 65
    user_values = [n, p, k]
    benchmark_values = [104.0, 50.0, 65.0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=benchmark_values + [benchmark_values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(116, 198, 157, 0.25)",
        line=dict(color="#52b788", dash="dash", width=2),
        name="Dataset Median Benchmark"
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=user_values + [user_values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(27, 67, 50, 0.45)",
        line=dict(color="#1b4332", width=3),
        name="Current Soil Profile"
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(max(user_values), max(benchmark_values)) * 1.25], gridcolor=PALETTE["grid_color"]),
            angularaxis=dict(gridcolor=PALETTE["grid_color"])
        ),
        font=dict(family="Plus Jakarta Sans, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Interactive correlation matrix heatmap for numeric features."""
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0.0, "#d8f3dc"],
            [0.5, "#ffffff"],
            [1.0, "#1b4332"]
        ],
        zmin=-1,
        zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        hoverongaps=False
    ))
    
    fig.update_layout(
        BASE_LAYOUT,
        title="<b>Agronomic Feature Correlation Matrix</b>",
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange="reversed")
    )
    return fig
