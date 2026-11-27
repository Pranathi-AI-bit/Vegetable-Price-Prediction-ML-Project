
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score
)


# Model	ML type	Evaluation
# Random Forest Regressor	Regression	MAE, RMSE, R²
# Extra Trees Regressor	Regression	MAE, RMSE, R²
# Random Forest Classifier	Classification	Accuracy, F1
# Extra Trees Classifier	Classification	Accuracy, F1
# 1. Load dataset
df = pd.read_csv(
    "data/feature_engineered_vegetable_prices.csv",
    parse_dates=["Arrival_Date"]
)

# 2. Convert historical prices to ₹/kg
df["Lag_1"] = df["Lag_1"] / 100
df["Rolling_7_Mean"] = df["Rolling_7_Mean"] / 100

# 3. Load saved models
reg_bundle = joblib.load("models/regression_model.pkl")
cls_bundle = joblib.load("models/classification_model.pkl")

reg_model = reg_bundle["model"]
cls_model = cls_bundle["model"]

feature_columns = reg_bundle["feature_columns"]

# 4. Prepare chronological test data
unique_dates = sorted(df["Arrival_Date"].unique())
split_index = int(len(unique_dates) * 0.80)
split_date = unique_dates[split_index]

test_df = df[df["Arrival_Date"] >= split_date].copy()

X_test = test_df[feature_columns]

# 5. Regression evaluation
y_reg_test = test_df["Modal_Price_Per_Kg"]

reg_predictions = reg_model.predict(X_test)

reg_mae = mean_absolute_error(
    y_reg_test,
    reg_predictions
)

reg_rmse = np.sqrt(
    mean_squared_error(y_reg_test, reg_predictions)
)

reg_r2 = r2_score(
    y_reg_test,
    reg_predictions
)

# 6. Classification evaluation
thresholds = cls_bundle["thresholds"]

def classify_price(row):
    limits = thresholds[row["Commodity"]]
    price = row["Modal_Price_Per_Kg"]

    if price <= limits["low"]:
        return "Low"
    elif price <= limits["high"]:
        return "Medium"
    else:
        return "High"

y_cls_test = test_df.apply(
    classify_price,
    axis=1
)

cls_predictions = cls_model.predict(X_test)

cls_accuracy = accuracy_score(
    y_cls_test,
    cls_predictions
)

cls_f1 = f1_score(
    y_cls_test,
    cls_predictions,
    average="weighted"
)

# 7. Print results
print("\nREGRESSION MODEL EVALUATION")
print("Model:", reg_bundle["best_model_name"])
print("MAE:", round(reg_mae, 2))
print("RMSE:", round(reg_rmse, 2))
print("R2 Score:", round(reg_r2, 4))

print("\nCLASSIFICATION MODEL EVALUATION")
print("Model:", cls_bundle["best_model_name"])
print("Accuracy:", round(cls_accuracy, 4))
print("Weighted F1:", round(cls_f1, 4))

# 8. Regression actual vs predicted chart
plt.figure(figsize=(8, 5))
plt.scatter(y_reg_test, reg_predictions, alpha=0.3)
plt.xlabel("Actual Price (₹/kg)")
plt.ylabel("Predicted Price (₹/kg)")
plt.title("Actual vs Predicted Vegetable Prices")
plt.tight_layout()
plt.show()

# 9. Regression error chart
errors = y_reg_test.to_numpy() - reg_predictions

plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error (₹/kg)")
plt.ylabel("Frequency")
plt.title("Regression Prediction Errors")
plt.tight_layout()
plt.show()

# 10. Classification category chart
comparison = pd.DataFrame({
    "Actual": y_cls_test,
    "Predicted": cls_predictions
})

category_counts = comparison["Predicted"].value_counts()

plt.figure(figsize=(7, 5))
category_counts.reindex(
    ["Low", "Medium", "High"],
    fill_value=0
).plot(kind="bar")

plt.xlabel("Price Category")
plt.ylabel("Number of Predictions")
plt.title("Predicted Price Categories")
plt.tight_layout()
plt.show()

# 11. Save evaluation results
results = pd.DataFrame([
    {
        "Task": "Regression",
        "Best Model": reg_bundle["best_model_name"],
        "MAE": reg_mae,
        "RMSE": reg_rmse,
        "R2": reg_r2,
        "Accuracy": None,
        "Weighted F1": None
    },
    {
        "Task": "Classification",
        "Best Model": cls_bundle["best_model_name"],
        "MAE": None,
        "RMSE": None,
        "R2": None,
        "Accuracy": cls_accuracy,
        "Weighted F1": cls_f1
    }
])

results.to_csv(
    "models/model_evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved successfully!")
