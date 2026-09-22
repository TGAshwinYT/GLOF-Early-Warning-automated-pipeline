"""
Stage 1: Glacial Lake Breach Trigger Predictor (LightGBM / Tabular Time-Series).

Predicts:
1. Breach probability P(breach)
2. Peak breach discharge Q_peak (m3/s)

Input Features:
- 14-day cumulative degree-day melt (ERA5-Land)
- 48-hour antecedent rainfall (MOSDAC INSAT-3D)
- Lake surface area expansion rate Delta_A (%)
- Upstream hanging glacier / rock slope angle (degrees)
- Moraine dam deformation velocity (InSAR Sentinel-1 in mm/yr)
- Cascading lake proximity index [0, 1] (HydroSHEDS/Bhuvan cluster)
"""

import os
import argparse
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

class BreachPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.regressor = None
        self.classifier = None
        self.feature_names = [
            "cumulative_melt_14d",
            "rainfall_48h",
            "lake_area_change_pct",
            "slope_angle_deg",
            "moraine_creep_velocity_mmyr",
            "cascading_proximity_index"
        ]

    def generate_training_data(self, n_samples=2000, random_seed=42):
        """
        Generates synthetic training distribution based on empirical Himalayan GLOF statistics
        (Thame 2024, South Lhonak 2023, Chamoli 2021, Dig Tsho, Lemthang Tsho).
        """
        np.random.seed(random_seed)

        # 14-day melt: 20 to 180 degree-days (heatwaves spike to 120-180)
        melt = np.random.uniform(20.0, 180.0, n_samples)

        # 48-hour rainfall: 0 to 250 mm (monsoon downpours)
        rainfall = np.random.exponential(scale=35.0, size=n_samples)

        # Lake surface expansion: -2% to +35%
        area_change = np.random.normal(loc=5.0, scale=8.0, size=n_samples)

        # Slope angle: 15 to 65 degrees
        slope = np.random.uniform(15.0, 65.0, n_samples)

        # Moraine creep velocity: 2 to 75 mm/yr
        moraine_creep = np.random.uniform(2.0, 75.0, n_samples)

        # Cascading proximity index: 0.0 (isolated) to 1.0 (immediate upstream cascade)
        cascade_prox = np.random.beta(a=1.5, b=3.0, size=n_samples)

        # Physics-informed instability score
        # Thame effect: cascading lake proximity + heatwave melt + rain multiplicatively destabilizes dam
        # Chamoli effect: extreme rock/ice slope (>45 deg) triggers catastrophic avalanche surge
        instability_index = (
            0.20 * (melt / 180.0) +
            0.25 * (rainfall / 150.0) +
            0.20 * (moraine_creep / 75.0) +
            0.25 * (cascade_prox * 1.5) +
            0.15 * np.maximum(0.0, area_change / 30.0) +
            0.25 * np.maximum(0.0, (slope - 35.0) / 30.0) +
            np.random.normal(0, 0.05, n_samples)
        )

        # Probability of breach
        prob_breach = 1.0 / (1.0 + np.exp(-12.0 * (instability_index - 0.58)))
        breach_label = (prob_breach > 0.50).astype(int)

        # Peak discharge Q_peak (m3/s) based on Costa-Schuster / Froehlich dam breach hydraulics
        # Q_peak typically scales with lake volume & breach depth
        base_volume_factor = np.random.uniform(500.0, 2500.0, n_samples)
        q_peak = np.where(
            breach_label == 1,
            base_volume_factor * (1.0 + 1.2 * cascade_prox) * (1.0 + 0.8 * (rainfall / 100.0)),
            np.random.uniform(50.0, 250.0, n_samples) # normal non-breach base discharge
        )
        q_peak = np.clip(q_peak, 50.0, 12000.0)

        df = pd.DataFrame({
            "cumulative_melt_14d": melt,
            "rainfall_48h": rainfall,
            "lake_area_change_pct": area_change,
            "slope_angle_deg": slope,
            "moraine_creep_velocity_mmyr": moraine_creep,
            "cascading_proximity_index": cascade_prox,
            "breach_label": breach_label,
            "q_peak": q_peak
        })
        return df

    def train(self, df=None):
        if df is None:
            df = self.generate_training_data()

        X = df[self.feature_names]
        y_clf = df["breach_label"]
        y_reg = df["q_peak"]

        # Train Classifier (Breach Occurrence)
        X_train, X_val, y_clf_tr, y_clf_val = train_test_split(X, y_clf, test_size=0.2, random_state=42)
        self.classifier = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        self.classifier.fit(X_train, y_clf_tr)
        val_preds_prob = self.classifier.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_clf_val, val_preds_prob)
        print(f"[+] Stage 1 Classifier Trained: Validation ROC-AUC = {auc:.4f}")

        # Train Regressor (Peak Discharge Q_peak)
        # Train on breach instances
        breach_mask = df["breach_label"] == 1
        X_breach = X[breach_mask]
        y_breach = y_reg[breach_mask]
        X_tr_r, X_val_r, y_tr_r, y_val_r = train_test_split(X_breach, y_breach, test_size=0.2, random_state=42)

        self.regressor = lgb.LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
        self.regressor.fit(X_tr_r, y_tr_r)
        val_preds_q = self.regressor.predict(X_val_r)
        mae = mean_absolute_error(y_val_r, val_preds_q)
        r2 = r2_score(y_val_r, val_preds_q)
        print(f"[+] Stage 1 Regressor Trained: Validation MAE = {mae:.2f} m3/s, R2 = {r2:.4f}")

        # Serialize models
        clf_path = os.path.join(self.model_dir, "breach_classifier.joblib")
        reg_path = os.path.join(self.model_dir, "breach_regressor.joblib")
        joblib.dump(self.classifier, clf_path)
        joblib.dump(self.regressor, reg_path)
        print(f"[+] Serialized Stage 1 models to {self.model_dir}/")

    def predict(self, input_features: dict):
        """
        Takes raw feature dictionary and predicts breach probability & Q_peak.
        """
        if self.classifier is None or self.regressor is None:
            self.classifier = joblib.load(os.path.join(self.model_dir, "breach_classifier.joblib"))
            self.regressor = joblib.load(os.path.join(self.model_dir, "breach_regressor.joblib"))

        input_df = pd.DataFrame([input_features])[self.feature_names]
        prob_breach = float(self.classifier.predict_proba(input_df)[0, 1])
        pred_q_peak = float(self.regressor.predict(input_df)[0]) if prob_breach > 0.35 else 50.0

        return {
            "breach_probability": prob_breach,
            "predicted_q_peak_m3s": pred_q_peak,
            "alert_level": "CRITICAL" if prob_breach > 0.70 else ("WARNING" if prob_breach > 0.40 else "NORMAL")
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Glacial Lake Breach Predictor")
    parser.add_argument("--train", action="store_true", default=True, help="Train models")
    args = parser.parse_args()

    predictor = BreachPredictor()
    predictor.train()
