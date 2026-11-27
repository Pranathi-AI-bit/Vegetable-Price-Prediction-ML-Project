
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# Metric	Meaning
# Accuracy	Total predictions-la correct predictions percentage
# Precision	Model predict panna category-la correct percentage
# Recall	Actual category records-la correctly identified percentage
# F1-score	Precision and Recall balance
# Confusion Matrix	Correct and wrong category predictions
# 1. Load dataset
df = pd.read_csv(
    "data/feature_engineered_vegetable_prices.csv",
    parse_dates=["Arrival_Date"]
)

# 2. Convert historical prices to ₹/kg
df["Lag_1"] = df["Lag_1"] / 100
df["Rolling_7_Mean"] = df["Rolling_7_Mean"] / 100

# 3. Define input features
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

# 4. Chronological train-test split
unique_dates = sorted(df["Arrival_Date"].unique())

split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

train_mask = df["Arrival_Date"] < split_date
test_mask = df["Arrival_Date"] >= split_date

train_df = df.loc[train_mask].copy()
test_df = df.loc[test_mask].copy()

# 5. Create price categories using training data only
thresholds = {}

for commodity in train_df["Commodity"].unique():

    commodity_prices = train_df.loc[
        train_df["Commodity"] == commodity,
        "Modal_Price_Per_Kg"
    ]

    low_limit = commodity_prices.quantile(0.33)
    high_limit = commodity_prices.quantile(0.67)

    thresholds[commodity] = {
        "low": float(low_limit),
        "high": float(high_limit)
    }


def classify_price(row):

    limits = thresholds[row["Commodity"]]
    price = row["Modal_Price_Per_Kg"]

    if price <= limits["low"]:
        return "Low"

    elif price <= limits["high"]:
        return "Medium"

    else:
        return "High"


# 6. Create target categories
train_df["Price_Category"] = train_df.apply(
    classify_price,
    axis=1
)

test_df["Price_Category"] = test_df.apply(
    classify_price,
    axis=1
)

X_train = train_df[feature_columns]
X_test = test_df[feature_columns]

y_train = train_df["Price_Category"]
y_test = test_df["Price_Category"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Test starts from:", split_date)

# 7. Preprocessing
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

# 8. Classification models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
}

results = []
best_model = None
best_model_name = None
best_f1 = -1

# 9. Train and evaluate
for name, classifier in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier)
        ]
    )

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["Low", "Medium", "High"],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["Low", "Medium", "High"]
        )
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Weighted_F1": weighted_f1
    })

    if weighted_f1 > best_f1:
        best_f1 = weighted_f1
        best_model = pipeline
        best_model_name = name

# 10. Compare models
results_df = pd.DataFrame(results)

print("\nCLASSIFICATION MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nBEST CLASSIFIER:", best_model_name)

# 11. Save model
Path("models").mkdir(exist_ok=True)

model_bundle = {
    "model": best_model,
    "feature_columns": feature_columns,
    "best_model_name": best_model_name,
    "thresholds": thresholds,
    "classes": ["Low", "Medium", "High"],
    "split_date": str(split_date)
}

joblib.dump(
    model_bundle,
    "models/classification_model.pkl"
)

print("\nClassification model saved successfully!")
