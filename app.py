"""CropYield AI - Premium Agricultural Machine Learning Platform.
Main Application Entrypoint and Multi-Page Router.
"""

import os
import streamlit as st

# Configure wide layout, title, and favicon
st.set_page_config(
    page_title="CropYield AI — Agricultural Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and inject custom CSS stylesheet
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "styles", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Import view components
from views.overview import render_overview
from views.dataset_explorer import render_dataset_explorer
from views.model_performance import render_model_performance
from views.yield_prediction import render_yield_prediction
from views.crop_recommendation import render_crop_recommendation
from views.smart_insights import render_smart_insights
from views.prediction_history import render_prediction_history
from views.about import render_about
from utils.model_engine import load_all_models
from utils.history_manager import init_history

# Initialize session history
init_history()

# Verify models loading at startup
try:
    models = load_all_models()
    models_status = "All 4 Models Active"
    status_color = "#52b788"
except Exception as e:
    models_status = "Model Loading Issue"
    status_color = "#f87171"

# Sidebar Navigation and Branding
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.25rem 0.5rem 1rem 0.5rem;">
        <div style="font-size: 2.8rem; margin-bottom: 0.25rem;">🌾</div>
        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">
            CropYield <span style="color: #52b788;">AI</span>
        </div>
        <div style="font-size: 0.8rem; color: #a7f3d0; margin-top: 2px; font-weight: 500;">
            Agricultural Intelligence SaaS
        </div>
        <div style="display: inline-block; margin-top: 8px; padding: 3px 10px; background: rgba(82, 183, 136, 0.2); border: 1px solid rgba(82, 183, 136, 0.4); border-radius: 9999px; font-size: 0.72rem; color: #d8f3dc; font-weight: 600;">
            v2.4 Pro • Scikit-learn 1.6.1
        </div>
    </div>
    <hr style="margin: 0.5rem 0 1rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.12);">
    """, unsafe_allow_html=True)
    
    # Navigation items
    page_options = [
        "🌾 Overview Dashboard",
        "🔍 Dataset Explorer",
        "📊 Model Performance",
        "🌱 Crop Yield Prediction",
        "💡 Crop Recommendation",
        "🧠 Smart Farming Insights",
        "📜 Prediction History",
        "ℹ️ About Project"
    ]
    
    selected_page = st.radio(
        "Navigation Menu",
        page_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("""
    <hr style="margin: 1.25rem 0 1rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.12);">
    <div style="padding: 0 0.5rem;">
        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; font-weight: 700; margin-bottom: 0.5rem;">
            SYSTEM STATUS
        </div>
        <div style="background: rgba(0, 0, 0, 0.25); border-radius: 8px; padding: 10px; border: 1px solid rgba(255, 255, 255, 0.08);">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 0.8rem; color: #e2e8f0;">ML Pipelines</span>
                <span style="display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; color: #52b788; font-weight: 600;">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background: #52b788; display: inline-block;"></span>
                    Operational
                </span>
            </div>
            <div style="font-size: 0.74rem; color: #94a3b8; line-height: 1.4;">
                • Random Forest (Champion)<br>
                • Linear Regression<br>
                • SVR (RBF Kernel)<br>
                • Decision Tree Regressor
            </div>
        </div>
        
        <div style="margin-top: 1.25rem; font-size: 0.72rem; color: #64748b; text-align: center;">
            CropYield AI • Academic ML Project<br>
            Multi-Model Agricultural Engine
        </div>
    </div>
    """, unsafe_allow_html=True)

# Page Routing
if selected_page == "🌾 Overview Dashboard":
    render_overview()
elif selected_page == "🔍 Dataset Explorer":
    render_dataset_explorer()
elif selected_page == "📊 Model Performance":
    render_model_performance()
elif selected_page == "🌱 Crop Yield Prediction":
    render_yield_prediction()
elif selected_page == "💡 Crop Recommendation":
    render_crop_recommendation()
elif selected_page == "🧠 Smart Farming Insights":
    render_smart_insights()
elif selected_page == "📜 Prediction History":
    render_prediction_history()
elif selected_page == "ℹ️ About Project":
    render_about()
