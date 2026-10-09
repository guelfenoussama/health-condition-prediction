import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify





# ==============================================================================
# CLASS: AutoTabularFeatures
# ==============================================================================
# This class acts as the core feature engineering engine for the inference pipeline.
# When a raw request payload (single patient dictionary or batch of patients) arrives 
# via the Flask API, it does not match the exact feature set expected by the model. 
# 
# To bridge this gap, this class reconstructs all complex tabular features originally 
# engineered during the notebook training phase. Specifically, it executes:
# 1. Feature Interactions & Ratios: Calculates row-level summary statistics (mean, std, 
#    min, max) across numerical attributes and creates pairwise products and safe ratios.
# 2. Group Aggregations: Applies pre-computed group-level metrics (mean, std, diff) 
#    using historical summary statistics stored inside the trained pipeline object.
# 3. Categorical Encodings: Maps incoming string categories using stored frequency 
#    maps and label maps, while gracefully handling missing categories or unobserved values.
# 4. Imputation & Alignment: Fills missing values with training medians and reindexes 
#    the resulting DataFrame to enforce exact feature column ordering and data types.
# ==============================================================================



class AutoTabularFeatures:
    def __init__(self):
        self.raw_columns = None
        self.drop_columns = []
        self.pair_columns = []
        self.group_specs = []
        self.cat_columns = []
        self.freq_maps = {}
        self.label_maps = {}
        self.medians = {}
        self.output_columns = []

    @staticmethod
    def _safe_ratio(a, b):
        return a.astype("float32") / b.astype("float32").replace(0, np.nan)

    @staticmethod
    def _clean(frame):
        return frame.replace([np.inf, -np.inf], np.nan)

    def _add_interactions(self, frame):
        frame = frame.copy()
        numeric_cols = frame.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) >= 2:
            numeric_block = frame[numeric_cols]
            frame["fe_numeric_row_mean"] = numeric_block.mean(axis=1)
            frame["fe_numeric_row_std"] = numeric_block.std(axis=1)
            frame["fe_numeric_row_min"] = numeric_block.min(axis=1)
            frame["fe_numeric_row_max"] = numeric_block.max(axis=1)

        for col_a, col_b in self.pair_columns:
            if col_a not in frame.columns or col_b not in frame.columns:
                continue
            name_a = str(col_a)
            name_b = str(col_b)
            frame[f"fe_product__{name_a}__{name_b}"] = (
                frame[col_a].astype("float32") * frame[col_b].astype("float32")
            )
            frame[f"fe_ratio__{name_a}__{name_b}"] = self._safe_ratio(
                frame[col_a], frame[col_b]
            )
            frame[f"fe_ratio__{name_b}__{name_a}"] = self._safe_ratio(
                frame[col_b], frame[col_a]
            )

        return self._clean(frame)

    def _apply_group_features(self, frame):
        for group_col, numeric_col, stats in self.group_specs:
            if group_col not in frame.columns or numeric_col not in frame.columns:
                continue
            groups = frame[group_col].astype(str).fillna("__MISSING__")
            feature_prefix = f"fe_group__{group_col}__{numeric_col}"

            frame[f"{feature_prefix}__mean"] = groups.map(stats["mean"])
            frame[f"{feature_prefix}__std"] = groups.map(stats["std"])
            frame[f"{feature_prefix}__diff"] = (
                frame[numeric_col] - frame[f"{feature_prefix}__mean"]
            )

        return frame

    def transform_test(self, test_frame):
        frame = test_frame.reindex(columns=self.raw_columns).copy()
        frame = frame.drop(columns=self.drop_columns, errors="ignore")

        frame = self._add_interactions(frame)
        frame = self._apply_group_features(frame)

        for col in self.cat_columns:
            if col not in frame.columns:
                frame[col] = "__MISSING__"

            values = frame[col].astype(str).fillna("__MISSING__")

            frame[f"fe_frequency__{col}"] = values.map(
                self.freq_maps[col]
            ).fillna(0).astype("float32")

            frame[col] = values.map(
                self.label_maps[col]
            ).fillna(-1).astype("int32")

        for col, median in self.medians.items():
            if col in frame.columns:
                frame[col] = frame[col].replace(
                    [np.inf, -np.inf], np.nan
                ).fillna(median)

        return (
            frame.reindex(columns=self.output_columns, fill_value=0)
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .astype("float32")
        )
# Class definition for custom Weighted Ensemble model


# ==============================================================================
# CLASS: WeightedEnsemble & Pickle Unpickling Hook
# ==============================================================================
# This class defines the custom ensemble architecture that blends probabilistic 
# predictions from both XGBoost and LightGBM models using pre-tuned blending weights.
#
# During prediction (`predict_proba`), it queries both individual classifiers, multiplies 
# their output class probability matrices by their respective ensemble weights (0.45 / 0.55),
# and sums them up to yield a robust, calibrated final probability distribution.
#
# TECHNICAL NOTE ON UNPICKLING:
# When model artifacts were serialized using `pickle` inside the Jupyter Notebook, 
# Python recorded the class definitions under the default `__main__` namespace. 
# To enable Gunicorn and Flask to successfully deserialize (unpickle) `model.bin` without 
# throwing an `AttributeError: Can't get attribute ...`, we explicitly alias both 
# `WeightedEnsemble` and `AutoTabularFeatures` back into the `__main__` module scope prior 
# to executing `pickle.load()`.
# ==============================================================================

class WeightedEnsemble:
    __module__ = '__main__'
    def __init__(self, xgb_model, lgb_model, weights=(0.45, 0.55)):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.weights = np.asarray(weights, dtype=float) / np.sum(weights)

    def predict_proba(self, X):
        return (
            self.weights[0] * self.xgb_model.predict_proba(X)
            + self.weights[1] * self.lgb_model.predict_proba(X)
        )
# Alias main module to ensure pickle unpickling compatibility with Gunicorn
import __main__
__main__.WeightedEnsemble = WeightedEnsemble
__main__.AutoTabularFeatures = AutoTabularFeatures

# Load model artifacts from serialized pickle file
MODEL_FILE = "model.bin"
with open(MODEL_FILE, "rb") as f_in:
    artifacts = pickle.load(f_in)

model = artifacts["model"]
y_encoder = artifacts["y_encoder"]
feature_builder = artifacts["feature_builder"]
variance_selector = artifacts["variance_selector"]


# Initialize Flask application
app = Flask("health-condition-prediction")


# Prediction endpoint definition
@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        if isinstance(input_data, dict):
            df_raw = pd.DataFrame([input_data])
        else:
            df_raw = pd.DataFrame(input_data)

        # 1. Apply automated feature engineering pipeline
        df_engineered = feature_builder.transform_test(df_raw)

        # 2. Apply variance threshold selector and clean infinite/NaN values
        df_engineered = df_engineered.replace([np.inf, -np.inf], 0).fillna(0)
        X_transformed = variance_selector.transform(df_engineered).astype("float32")

        # 3. Predict class probabilities and decode output label
        probabilities = model.predict_proba(X_transformed)
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_labels = y_encoder.inverse_transform(predicted_indices)
        max_probabilities = np.max(probabilities, axis=1)

        result = {
            "prediction": predicted_labels[0] if len(predicted_labels) == 1 else predicted_labels.tolist(),
            "probability": float(max_probabilities[0]) if len(max_probabilities) == 1 else max_probabilities.tolist()
        }

        return jsonify(result)

    except Exception as e:
        # Print stack trace error for container logging and return 500 status code
        print(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)