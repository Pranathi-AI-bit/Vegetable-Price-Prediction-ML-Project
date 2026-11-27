
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR / "data" / "featured_vegetable_prices.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. FEATURES AND TARGET
# =====================================

FEATURES = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "DayOfYear",
    "Lag_1",
    "Rolling_7_Mean",
    "Rolling_30_Mean"
]

TARGET = "Modal_Price"


# =====================================
# 3. TRAINING FUNCTION
# =====================================

def train_models():

    print("Loading featured dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Featured dataset not found. "
            "Run python feature_engineering.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    # Convert date
    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )

    # Check columns
    missing = [
        col for col in FEATURES + [TARGET, "Arrival_Date"]
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Remove rows without historical information
    df = df.dropna(
        subset=FEATURES + [TARGET, "Arrival_Date"]
    )

    if len(df) < 20:
        raise ValueError(
            f"Only {len(df)} valid records available. "
            "At least 20 records are required for this training demo. "
            "Collect more historical market data."
        )

    # =====================================
    # 4. SORT CHRONOLOGICALLY
    # =====================================

    df = df.sort_values(
        "Arrival_Date"
    ).reset_index(drop=True)

    X = df[FEATURES]
    y = df[TARGET]

    # =====================================
    # 5. CHRONOLOGICAL TRAIN TEST SPLIT
    # =====================================

    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nDataset Information")
    print("Total records:", len(df))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # =====================================
    # 6. PREPROCESSING
    # =====================================

    categorical_features = [
        "State",
        "District",
        "Market",
        "Commodity"
    ]

    numerical_features = [
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "DayOfYear",
        "Lag_1",
        "Rolling_7_Mean",
        "Rolling_30_Mean"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    # =====================================
    # 7. MACHINE LEARNING MODELS
    # =====================================

    models = {

        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    best_model = None
    best_model_name = None
    best_mae = float("inf")

    # =====================================
    # 8. TRAIN AND EVALUATE
    # =====================================

    for name, algorithm in models.items():

        print(f"\nTraining {name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", algorithm)
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        print("MAE:", round(mae, 2))
        print("RMSE:", round(rmse, 2))
        print("R2 Score:", round(r2, 4))

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })

        # Select model based on lowest MAE
        if mae < best_mae:

            best_mae = mae
            best_model = pipeline
            best_model_name = name

        model_path = MODEL_DIR / f"{name.lower()}_model.pkl"

        joblib.dump(
            pipeline,
            model_path
        )

        print("Model saved:", model_path)

    # =====================================
    # 9. SAVE MODEL COMPARISON
    # =====================================

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        MODEL_DIR / "model_comparison.csv",
        index=False
    )

    print("\nMODEL COMPARISON")

    print(results_df.to_string(index=False))

    # =====================================
    # 10. SAVE BEST MODEL
    # =====================================

    best_model_path = MODEL_DIR / "best_model.pkl"

    joblib.dump(
        best_model,
        best_model_path
    )

    print("\nBest Model:", best_model_name)
    print("Best MAE:", round(best_mae, 2))
    print("Best model saved:", best_model_path)

    print("\nModel training completed!")


# =====================================
# 11. MAIN
# =====================================

if __name__ == "__main__":
    train_models()
