"""Model Engine for CropYield AI.
Provides cached loading of trained scikit-learn 1.6.1 model pipelines,
inference execution, feature importance extraction, and crop recommendation.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_loader import (
    load_clean_dataset,
    load_feature_columns,
    load_model_performance,
    CANONICAL_CROPS
)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATHS = {
    "Random Forest": os.path.join(_BASE_DIR, "random_forest.pkl"),
    "Decision Tree": os.path.join(_BASE_DIR, "decision_tree.pkl"),
    "Linear Regression": os.path.join(_BASE_DIR, "linear_regression.pkl"),
    "SVR": os.path.join(_BASE_DIR, "svr.pkl")
}


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load and cache all 4 trained machine learning pipelines."""
    models = {}
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            raise FileNotFoundError(f"Model file not found at: {path}")
    return models


def get_model(name: str):
    """Retrieve a specific cached model pipeline by display name."""
    models = load_all_models()
    return models.get(name, models["Random Forest"])


def format_input_dataframe(input_dict: dict) -> pd.DataFrame:
    """Format input dictionary into a single-row DataFrame with the exact feature order."""
    feature_cols = load_feature_columns()
    
    # Cast numeric columns appropriately
    numeric_keys = [
        "area_hectare", "rainfall_mm", "temperature_c", "humidity",
        "nitrogen", "phosphorus", "potassium", "fertilizer_used"
    ]
    formatted = {}
    for col in feature_cols:
        val = input_dict.get(col, None)
        if col in numeric_keys:
            formatted[col] = float(val) if val is not None else 0.0
        else:
            formatted[col] = str(val) if val is not None else ""
            
    df = pd.DataFrame([formatted], columns=feature_cols)
    return df


def predict_single(model_name: str, input_dict: dict) -> dict:
    """Run real inference on a single farm input record using the specified model."""
    model = get_model(model_name)
    input_df = format_input_dataframe(input_dict)
    
    raw_pred = float(model.predict(input_df)[0])
    # Agricultural yield cannot be physically negative
    predicted_yield = max(0.1, round(raw_pred, 3))
    
    area = float(input_dict.get("area_hectare", 1.0))
    total_production = round(predicted_yield * area, 2)
    
    # Classify yield level
    if predicted_yield < 2.5:
        category = "Low Yield (< 2.5 t/ha)"
        category_color = "#dc2626"
    elif predicted_yield <= 4.0:
        category = "Moderate Yield (2.5 - 4.0 t/ha)"
        category_color = "#f59e0b"
    else:
        category = "High Yield (> 4.0 t/ha)"
        category_color = "#16a34a"
        
    perf_df = load_model_performance()
    model_stats = perf_df[perf_df["Model"] == model_name].iloc[0] if model_name in perf_df["Model"].values else None
    
    r2_score_val = float(model_stats["R2 Score"]) if model_stats is not None else 0.618
    rmse_val = float(model_stats["RMSE"]) if model_stats is not None else 0.634
    
    return {
        "model_name": model_name,
        "predicted_yield": predicted_yield,
        "total_production": total_production,
        "area": area,
        "yield_category": category,
        "category_color": category_color,
        "r2_score": r2_score_val,
        "rmse": rmse_val,
        "lower_bound": max(0.0, round(predicted_yield - rmse_val, 2)),
        "upper_bound": round(predicted_yield + rmse_val, 2)
    }


def predict_all_models(input_dict: dict) -> pd.DataFrame:
    """Run real inference across all 4 models simultaneously for direct comparison."""
    models = load_all_models()
    perf_df = load_model_performance().set_index("Model")
    input_df = format_input_dataframe(input_dict)
    area = float(input_dict.get("area_hectare", 1.0))
    
    rows = []
    rf_pred = None
    
    # Calculate RF prediction first as benchmark
    if "Random Forest" in models:
        rf_pred = float(models["Random Forest"].predict(input_df)[0])
        
    for name, model in models.items():
        pred = max(0.1, float(model.predict(input_df)[0]))
        prod = round(pred * area, 2)
        r2 = perf_df.loc[name, "R2 Score"] if name in perf_df.index else np.nan
        rmse = perf_df.loc[name, "RMSE"] if name in perf_df.index else np.nan
        mae = perf_df.loc[name, "MAE"] if name in perf_df.index else np.nan
        
        diff_from_rf = (pred - rf_pred) if rf_pred is not None else 0.0
        
        rows.append({
            "Model": name,
            "Predicted Yield (t/ha)": round(pred, 3),
            "Total Production (Tons)": prod,
            "Variance vs RF (t/ha)": round(diff_from_rf, 3),
            "R² Score": round(r2, 4),
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        })
        
    return pd.DataFrame(rows)


def predict_batch(model_name: str, uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """Run batch prediction on an uploaded DataFrame."""
    model = get_model(model_name)
    feature_cols = load_feature_columns()
    
    # Verify missing columns
    missing = [c for c in feature_cols if c not in uploaded_df.columns]
    if missing:
        raise ValueError(f"Uploaded file missing required columns: {', '.join(missing)}")
        
    df_eval = uploaded_df[feature_cols].copy()
    preds = model.predict(df_eval)
    
    result_df = uploaded_df.copy()
    result_df["predicted_yield_ton_per_ha"] = np.clip(np.round(preds, 3), 0.1, None)
    
    if "area_hectare" in result_df.columns:
        result_df["total_production_tons"] = np.round(
            result_df["predicted_yield_ton_per_ha"] * result_df["area_hectare"], 2
        )
        
    return result_df


@st.cache_data(show_spinner=False)
def get_feature_importances() -> pd.DataFrame:
    """Extract real feature importances from the trained Random Forest pipeline."""
    rf = get_model("Random Forest")
    preprocessor = rf.named_steps["preprocessing"]
    model = rf.named_steps["model"]
    
    cat_encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    cat_cols = preprocessor.transformers[1][2]
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
    
    num_cols = preprocessor.transformers[0][2]
    all_feature_names = list(num_cols) + list(encoded_cat_names)
    
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    # Calculate percentage
    fi_df["Importance %"] = np.round(fi_df["Importance"] * 100, 2)
    return fi_df


@st.cache_data(show_spinner=False)
def get_test_predictions():
    """Generate true test-set predictions (20% split, random_state=42) matching model training."""
    df = load_clean_dataset()
    target_col = "yield_ton_per_hectare"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    models = load_all_models()
    test_eval_df = X_test.copy()
    test_eval_df["Actual_Yield"] = y_test
    
    for name, model in models.items():
        col_name = f"Pred_{name.replace(' ', '_')}"
        test_eval_df[col_name] = model.predict(X_test)
        
    return test_eval_df


def recommend_crops(base_env: dict) -> pd.DataFrame:
    """Intelligently rank all candidate crops for a given set of soil and climate conditions."""
    rf_model = get_model("Random Forest")
    feature_cols = load_feature_columns()
    
    recommendations = []
    
    for crop in CANONICAL_CROPS:
        crop_input = base_env.copy()
        crop_input["crop"] = crop
        
        # Predict yield with RF model
        input_df = format_input_dataframe(crop_input)
        pred_yield = float(rf_model.predict(input_df)[0])
        pred_yield = max(0.1, round(pred_yield, 3))
        
        # Determine crop agronomic profile
        if crop in ["Sugarcane", "Potato"]:
            historical_max = 5.8
        elif crop in ["Rice", "Wheat"]:
            historical_max = 4.8
        elif crop in ["Maize"]:
            historical_max = 4.2
        else:
            historical_max = 3.5
            
        # Suitability percentage (0-100%)
        suitability_pct = min(98.5, max(45.0, round((pred_yield / historical_max) * 100, 1)))
        
        # Agronomic advisory
        if crop == "Rice":
            tip = "High water demand. Monitor submergence depth and maintain consistent standing water during panicle initiation."
        elif crop == "Wheat":
            tip = "Requires cool canopy temperature during grain filling. Crown root irrigation (21 days post sowing) is critical."
        elif crop == "Sugarcane":
            tip = "Heavy nutrient consumer. Split nitrogen application into 3-4 doses to maximize cane diameter and sucrose content."
        elif crop == "Potato":
            tip = "Prefers well-drained friable soil. Avoid waterlogging to prevent tuber rotting and late blight infection."
        elif crop == "Maize":
            tip = "Highly responsive to nitrogen. Ensure good aeration and weed control during early knee-high vegetative stage."
        elif crop == "Mustard":
            tip = "Low moisture requirement. Susceptible to aphid infestation; apply balanced sulfur for enhanced oil content."
        elif crop == "Cotton":
            tip = "Deep taproot system; benefits from black soil moisture retention. Guard against bollworm and square drop."
        else:
            tip = "Maintain balanced macro-nutrients (N-P-K) and periodic soil testing."
            
        recommendations.append({
            "Crop": crop,
            "Expected Yield (t/ha)": pred_yield,
            "Suitability Score (%)": suitability_pct,
            "Water Need": "High" if crop in ["Rice", "Sugarcane"] else ("Medium" if crop in ["Wheat", "Potato", "Maize"] else "Low"),
            "Key Agronomic Tip": tip
        })
        
    rec_df = pd.DataFrame(recommendations).sort_values("Expected Yield (t/ha)", ascending=False).reset_index(drop=True)
    rec_df["Rank"] = range(1, len(rec_df) + 1)
    return rec_df
