
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

# Metric -	Meaning	Best result
# MAE -	Average prediction error	Lower
# RMSE -	Penalizes larger errors	Lower
# R² Score -	How well model explains price variation	Closer to 1

# 1. Load feature-engineered dataset
df = pd.read_csv(
    "data/feature_engineered_vegetable_prices.csv",
    parse_dates=["Arrival_Date"]
)

# 2. Convert historical prices into rupees per kg
df["Lag_1"] = df["Lag_1"] / 100
df["Rolling_7_Mean"] = df["Rolling_7_Mean"] / 100

# 3. Select input features
categorical_features = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade"
]

numerical_features = [
    "Year",
    "Month",
    "Day",
    "Day_of_Week",
    "Day_of_Year",
    "Lag_1",
    "Rolling_7_Mean"
]

feature_columns = categorical_features + numerical_features

# 4. Select target
target = "Modal_Price_Per_Kg"

X = df[feature_columns]
y = df[target]

# 5. Chronological train-test split
unique_dates = sorted(df["Arrival_Date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_mask = df["Arrival_Date"] < split_date
test_mask = df["Arrival_Date"] >= split_date

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Test starts from:", split_date)

# 6. Preprocessing categorical columns
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

# 7. Define regression models
models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_model_name = None
best_mae = float("inf")

# 8. Train and evaluate models
for name, regressor in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    # Select model with lowest MAE
    if mae < best_mae:
        best_mae = mae
        best_model = pipeline
        best_model_name = name

# 9. Display model comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nBEST MODEL:", best_model_name)

# 10. Save best model
Path("models").mkdir(exist_ok=True)

model_bundle = {
    "model": best_model,
    "feature_columns": feature_columns,
    "best_model_name": best_model_name,
    "target": target,
    "split_date": str(split_date)
}

joblib.dump(
    model_bundle,
    "models/regression_model.pkl"
)

print("\nRegression model saved successfully!")
