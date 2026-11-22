"""Data Loader module for CropYield AI dashboard.
Handles cached loading of cleaned and messy datasets, feature schemas, and performance benchmarks.
"""

import pandas as pd
import streamlit as st
import os

# Use paths relative to the project root (parent of utils/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_DATASET_PATH = os.path.join(_BASE_DIR, "crop_yield_cleaned (4).xlsx")
MESSY_DATASET_PATH = os.path.join(_BASE_DIR, "Crop_Yield_Messy_Dataset_500.xlsx")
FEATURE_COLUMNS_PATH = os.path.join(_BASE_DIR, "feature_columns.csv")
MODEL_PERFORMANCE_PATH = os.path.join(_BASE_DIR, "model_performance.csv")

# Canonical UI mappings
CANONICAL_CROPS = [
    "Cotton", "Maize", "Mustard", "Potato", "Rice", "Sugarcane", "Wheat"
]

CANONICAL_STATES = [
    "Bihar", "Karnataka", "Maharashtra", "Odisha", "Punjab", "West Bengal"
]

STATE_DISTRICT_MAP = {
    "Bihar": "Patna",
    "Karnataka": "Mysuru",
    "Maharashtra": "Pune",
    "Odisha": "Cuttack",
    "Punjab": "Ludhiana",
    "West Bengal": "Purulia"
}

SEASONS = ["Kharif", "Rabi", "Zaid"]
SOIL_TYPES = ["Black", "Clay", "Loamy", "Red", "Sandy"]
IRRIGATION_OPTIONS = ["Yes", "No"]

DATA_DICTIONARY = {
    "state": "Indian federal state where agricultural cultivation occurred.",
    "district": "Administrative district of cultivation location.",
    "crop": "Target agricultural crop variety planted.",
    "season": "Cultivation season (Kharif: Monsoon, Rabi: Winter, Zaid: Summer).",
    "area_hectare": "Total cultivated land area measured in hectares (ha).",
    "rainfall_mm": "Cumulative seasonal precipitation received in millimeters (mm).",
    "temperature_c": "Average ambient atmospheric temperature in degrees Celsius (°C).",
    "humidity": "Average relative atmospheric humidity percentage (%).",
    "soil_type": "Pedological classification of soil (Black, Clay, Loamy, Red, Sandy).",
    "nitrogen": "Soil available Nitrogen content index (kg/ha equivalent).",
    "phosphorus": "Soil available Phosphorus content index (kg/ha equivalent).",
    "potassium": "Soil available Potassium content index (kg/ha equivalent).",
    "fertilizer_used": "Total synthetic/organic fertilizer applied in kilograms (kg).",
    "irrigation": "Supplemental artificial water supply presence (Yes / No).",
    "yield_ton_per_hectare": "Recorded crop productivity harvested in metric tons per hectare."
}

PRESETS = {
    "Punjab Wheat (High-Yield Intensive)": {
        "state": "Punjab",
        "district": "Ludhiana",
        "crop": "Wheat",
        "season": "Rabi",
        "soil_type": "Loamy",
        "irrigation": "Yes",
        "area_hectare": 5.0,
        "rainfall_mm": 650.0,
        "temperature_c": 22.0,
        "humidity": 60.0,
        "nitrogen": 130.0,
        "phosphorus": 60.0,
        "potassium": 75.0,
        "fertilizer_used": 240.0
    },
    "Maharashtra Sugarcane (Cash Crop)": {
        "state": "Maharashtra",
        "district": "Pune",
        "crop": "Sugarcane",
        "season": "Rabi",
        "soil_type": "Black",
        "irrigation": "Yes",
        "area_hectare": 6.5,
        "rainfall_mm": 1200.0,
        "temperature_c": 30.0,
        "humidity": 72.0,
        "nitrogen": 150.0,
        "phosphorus": 70.0,
        "potassium": 90.0,
        "fertilizer_used": 280.0
    },
    "West Bengal Monsoon Rice": {
        "state": "West Bengal",
        "district": "Purulia",
        "crop": "Rice",
        "season": "Kharif",
        "soil_type": "Clay",
        "irrigation": "No",
        "area_hectare": 4.0,
        "rainfall_mm": 1850.0,
        "temperature_c": 31.0,
        "humidity": 85.0,
        "nitrogen": 110.0,
        "phosphorus": 50.0,
        "potassium": 60.0,
        "fertilizer_used": 190.0
    },
    "Karnataka Summer Maize": {
        "state": "Karnataka",
        "district": "Mysuru",
        "crop": "Maize",
        "season": "Zaid",
        "soil_type": "Red",
        "irrigation": "Yes",
        "area_hectare": 3.5,
        "rainfall_mm": 800.0,
        "temperature_c": 28.5,
        "humidity": 65.0,
        "nitrogen": 95.0,
        "phosphorus": 45.0,
        "potassium": 55.0,
        "fertilizer_used": 175.0
    }
}


@st.cache_data(show_spinner=False)
def load_clean_dataset() -> pd.DataFrame:
    """Load the primary cleaned agricultural dataset."""
    df = pd.read_excel(CLEAN_DATASET_PATH)
    # Ensure standard types and casing for consistent display
    return df


@st.cache_data(show_spinner=False)
def load_messy_dataset() -> pd.DataFrame:
    """Load the original messy dataset with duplicates and raw casing."""
    df = pd.read_excel(MESSY_DATASET_PATH)
    return df


@st.cache_data(show_spinner=False)
def load_feature_columns() -> list:
    """Load the list of feature column names used during model training."""
    df = pd.read_csv(FEATURE_COLUMNS_PATH, header=None)
    return df[0].tolist()


@st.cache_data(show_spinner=False)
def load_model_performance() -> pd.DataFrame:
    """Load the benchmark performance metrics computed on the 20% test split."""
    df = pd.read_csv(MODEL_PERFORMANCE_PATH)
    return df


def get_dataset_summary():
    """Compute high-level summary metrics for executive dashboard cards."""
    df = load_clean_dataset()
    models_df = load_model_performance()
    
    best_model_row = models_df.loc[models_df["R2 Score"].idxmax()]
    
    summary = {
        "total_records": len(df),
        "total_crops": len(CANONICAL_CROPS),
        "total_states": len(CANONICAL_STATES),
        "mean_yield": float(df["yield_ton_per_hectare"].mean()),
        "max_yield": float(df["yield_ton_per_hectare"].max()),
        "min_yield": float(df["yield_ton_per_hectare"].min()),
        "best_model_name": best_model_row["Model"],
        "best_model_r2": float(best_model_row["R2 Score"]),
        "best_model_rmse": float(best_model_row["RMSE"]),
        "models_count": len(models_df)
    }
    return summary
