"""About Project page for CropYield AI."""

import streamlit as st
import pandas as pd


def render_about():
    st.markdown("""
    <div class="page-banner">
        <h1>🌾 About CropYield AI — Precision Agricultural Intelligence</h1>
        <p>A machine learning-driven SaaS platform engineered to deliver accurate, interpretable, and actionable crop yield forecasting and agronomic decision support.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Abstract
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📄 Project Abstract & Problem Statement</div>
        <p style="color: #334155; line-height: 1.7; font-size: 0.95rem;">
            Agriculture in developing economies faces severe vulnerabilities stemming from erratic monsoon distributions, 
            rapid climatic shifts, and suboptimal fertilization practices. Traditional yield forecasting often relies on retrospective 
            surveys or rough rule-of-thumb heuristics that fail to capture the complex, non-linear interactions among meteorological, 
            pedological (soil), and management inputs.
        </p>
        <p style="color: #334155; line-height: 1.7; font-size: 0.95rem;">
            <b>CropYield AI</b> addresses this foundational challenge by operationalizing four distinct supervised machine learning 
            regressors trained on real multi-regional agronomic data across India. By standardizing 14 biological and environmental 
            parameters into an automated Scikit-learn preprocessing pipeline, the system delivers high-precision yield forecasts (t/ha), 
            crop suitability rankings, and soil diagnostic guidance before seed sowing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technical Architecture & Pipeline
    st.markdown("""
    <div class="section-card">
        <div class="section-title">⚙️ Machine Learning Pipeline & Data Engineering</div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; margin-top: 0.5rem;">
            <div style="background: #f8faf9; border: 1px solid #d8f3dc; border-radius: 10px; padding: 1.2rem;">
                <h4 style="color: #1b4332; margin-top: 0;">1. Data Preprocessing</h4>
                <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                    <b>Numerical Pipeline:</b> Handled via <code>SimpleImputer(strategy='median')</code> followed by <code>StandardScaler()</code> to eliminate feature scale disparities across rainfall (mm), temperature (°C), and nutrient indices (N-P-K).<br><br>
                    <b>Categorical Pipeline:</b> Handled via <code>SimpleImputer(strategy='most_frequent')</code> and <code>OneHotEncoder(handle_unknown='ignore')</code>, ensuring flawless runtime inference even for unexpected user inputs.
                </p>
            </div>
            <div style="background: #f8faf9; border: 1px solid #d8f3dc; border-radius: 10px; padding: 1.2rem;">
                <h4 style="color: #1b4332; margin-top: 0;">2. Regressor Architecture</h4>
                <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                    Four diverse algorithms evaluated on an identical 80/20 train/test split:
                    <br>• <b>Random Forest:</b> 200 estimators bagging ensemble (Champion, R² = 0.6177, RMSE = 0.6338).
                    <br>• <b>Linear Regression:</b> Parametric baseline (R² = 0.5113, RMSE = 0.7166).
                    <br>• <b>Support Vector Regression (SVR):</b> RBF kernel with C=10, epsilon=0.1 (R² = 0.4594).
                    <br>• <b>Decision Tree:</b> CART regressor with max_depth=8 (R² = 0.1774).
                </p>
            </div>
            <div style="background: #f8faf9; border: 1px solid #d8f3dc; border-radius: 10px; padding: 1.2rem;">
                <h4 style="color: #1b4332; margin-top: 0;">3. Evaluation & Validation</h4>
                <p style="font-size: 0.88rem; color: #475569; line-height: 1.6;">
                    Rigorous regression metrics verified across all pipelines:
                    <br>• <b>MAE (Mean Absolute Error):</b> Average physical tonnage variance.
                    <br>• <b>MSE (Mean Squared Error):</b> Penalizes large forecasting outliers.
                    <br>• <b>RMSE (Root Mean Squared Error):</b> Interpretable standard deviation of residuals.
                    <br>• <b>R² Score (Coefficient of Determination):</b> Proportion of variance explained.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Software Stack & System Specifications
    st.markdown("""
    <div class="section-card">
        <div class="section-title">💻 Technical Stack & Environment Specifications</div>
        <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem; color: #334155;">
            <tr style="border-bottom: 1px solid #e2e8f0; background: #f8faf9;">
                <th style="padding: 10px; text-align: left;">Component</th>
                <th style="padding: 10px; text-align: left;">Technology / Framework</th>
                <th style="padding: 10px; text-align: left;">Version / Configuration</th>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-weight: 600;">Core ML Library</td>
                <td style="padding: 10px;">Scikit-Learn</td>
                <td style="padding: 10px;"><code>1.6.1</code> (100% Model Pickle Compatibility)</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-weight: 600;">Frontend UI Framework</td>
                <td style="padding: 10px;">Streamlit</td>
                <td style="padding: 10px;"><code>1.67.0</code> (Wide layout, Custom Forest Green Theme)</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-weight: 600;">Interactive Visualizations</td>
                <td style="padding: 10px;">Plotly</td>
                <td style="padding: 10px;"><code>7.1.0</code> (Dynamic charts, radar gauges, heatmaps)</td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-weight: 600;">Data Manipulation</td>
                <td style="padding: 10px;">Pandas & NumPy</td>
                <td style="padding: 10px;"><code>3.1.1</code> / <code>2.4.6</code></td>
            </tr>
            <tr style="border-bottom: 1px solid #e2e8f0;">
                <td style="padding: 10px; font-weight: 600;">Model Serialization</td>
                <td style="padding: 10px;">Joblib</td>
                <td style="padding: 10px;"><code>1.6.0</code></td>
            </tr>
            <tr>
                <td style="padding: 10px; font-weight: 600;">Python Environment</td>
                <td style="padding: 10px;">CPython 3.11.18</td>
                <td style="padding: 10px;">Isolated virtual environment via <code>uv</code></td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # College Demonstration Guide Card
    st.markdown("""
    <div class="section-card">
        <div class="section-title">🎓 College Demonstration Walkthrough Guide</div>
        <p style="color: #334155; line-height: 1.6; font-size: 0.92rem;">
            When presenting this system to faculty, evaluators, or project reviewers, follow this recommended sequence:
        </p>
        <ol style="color: #334155; line-height: 1.8; font-size: 0.92rem;">
            <li><b>Overview Dashboard:</b> Introduce the agricultural problem, highlight the verified 500-sample dataset, and demonstrate the instant live inference estimator.</li>
            <li><b>Dataset Explorer:</b> Show the Raw vs. Cleaned comparison to prove robust data engineering (removal of 10 duplicate rows, median imputation of 25 nulls, snake_case normalization).</li>
            <li><b>Model Performance:</b> Walk through the benchmark leaderboard. Point out why Random Forest (R² = 0.6177) is the champion and demonstrate the live feature importance chart where rainfall accounts for 62.85% of weight.</li>
            <li><b>Crop Yield Prediction:</b> Load the "Punjab Wheat" or "Maharashtra Sugarcane" preset. Show how the model predicts yield, calculates total harvest tonnage, and compares all 4 models simultaneously.</li>
            <li><b>Crop Recommendation:</b> Show how the system recommends the most suitable crop for a farmer's specific soil and weather conditions.</li>
            <li><b>Smart Farming Insights:</b> Demo the N-P-K soil radar chart, climate sensitivity curve, and economic revenue projection.</li>
            <li><b>Prediction History:</b> Review the audit trail of past queries and demonstrate real-time CSV export.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
