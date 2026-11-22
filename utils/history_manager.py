"""Prediction History Manager for CropYield AI.
Provides persistent storage, export, and retrieval of inference queries.
"""

import os
from datetime import datetime
import pandas as pd
import streamlit as st

HISTORY_FILE = "a:/geni/prediction_history.csv"

COLUMNS = [
    "Timestamp", "State", "District", "Crop", "Season", "Area (ha)",
    "Rainfall (mm)", "Temp (°C)", "Soil Type", "Irrigation",
    "Model", "Predicted Yield (t/ha)", "Total Production (Tons)", "Yield Category"
]


def init_history():
    """Initialize history in session state and disk storage."""
    if "history_df" not in st.session_state:
        if os.path.exists(HISTORY_FILE):
            try:
                st.session_state.history_df = pd.read_csv(HISTORY_FILE)
            except Exception:
                st.session_state.history_df = pd.DataFrame(columns=COLUMNS)
        else:
            st.session_state.history_df = pd.DataFrame(columns=COLUMNS)


def add_prediction(input_dict: dict, prediction_result: dict):
    """Log a new prediction transaction to history."""
    init_history()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_entry = {
        "Timestamp": timestamp,
        "State": input_dict.get("state", "N/A"),
        "District": input_dict.get("district", "N/A"),
        "Crop": input_dict.get("crop", "N/A"),
        "Season": input_dict.get("season", "N/A"),
        "Area (ha)": float(input_dict.get("area_hectare", 1.0)),
        "Rainfall (mm)": float(input_dict.get("rainfall_mm", 0.0)),
        "Temp (°C)": float(input_dict.get("temperature_c", 0.0)),
        "Soil Type": input_dict.get("soil_type", "N/A"),
        "Irrigation": input_dict.get("irrigation", "N/A"),
        "Model": prediction_result.get("model_name", "Random Forest"),
        "Predicted Yield (t/ha)": prediction_result.get("predicted_yield", 0.0),
        "Total Production (Tons)": prediction_result.get("total_production", 0.0),
        "Yield Category": prediction_result.get("yield_category", "N/A")
    }
    
    entry_df = pd.DataFrame([new_entry])
    st.session_state.history_df = pd.concat([entry_df, st.session_state.history_df], ignore_index=True)
    
    # Persist to disk
    try:
        st.session_state.history_df.to_csv(HISTORY_FILE, index=False)
    except Exception as e:
        st.warning(f"Unable to write history file: {e}")


def get_history() -> pd.DataFrame:
    """Retrieve logged prediction records."""
    init_history()
    return st.session_state.history_df


def clear_history():
    """Clear all logged prediction records."""
    st.session_state.history_df = pd.DataFrame(columns=COLUMNS)
    if os.path.exists(HISTORY_FILE):
        try:
            os.remove(HISTORY_FILE)
        except Exception:
            pass


def get_history_stats() -> dict:
    """Calculate summary metrics over past predictions."""
    df = get_history()
    if df.empty:
        return {
            "total_runs": 0,
            "avg_predicted_yield": 0.0,
            "total_tonnage": 0.0,
            "top_crop": "None",
            "top_model": "None"
        }
        
    return {
        "total_runs": len(df),
        "avg_predicted_yield": round(float(df["Predicted Yield (t/ha)"].mean()), 2),
        "total_tonnage": round(float(df["Total Production (Tons)"].sum()), 2),
        "top_crop": df["Crop"].mode().iloc[0] if not df["Crop"].empty else "N/A",
        "top_model": df["Model"].mode().iloc[0] if not df["Model"].empty else "N/A"
    }
