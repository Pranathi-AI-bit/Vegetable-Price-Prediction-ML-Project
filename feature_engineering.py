
import pandas as pd
from pathlib import Path

# =====================================
# 1. PROJECT PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "cleaned_vegetable_prices.csv"
OUTPUT_FILE = BASE_DIR / "data" / "featured_vegetable_prices.csv"


# =====================================
# 2. FEATURE ENGINEERING FUNCTION
# =====================================

def create_features():

    print("Loading cleaned vegetable dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. "
            "Run python preprocessing.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Check required columns
    required_columns = [
        "State", "District", "Market",
        "Commodity", "Arrival_Date", "Modal_Price"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # =====================================
    # 3. DATA CLEANING
    # =====================================

    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )

    df["Modal_Price"] = pd.to_numeric(
        df["Modal_Price"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Arrival_Date", "Modal_Price"]
    )

    if df.empty:
        raise ValueError("No valid records found.")

    # =====================================
    # 4. SORT DATA
    # =====================================

    group_columns = [
        "State",
        "District",
        "Market",
        "Commodity"
    ]

    df = df.sort_values(
        group_columns + ["Arrival_Date"]
    ).reset_index(drop=True)

    # =====================================
    # 5. DATE FEATURES
    # =====================================

    df["Year"] = df["Arrival_Date"].dt.year

    df["Month"] = df["Arrival_Date"].dt.month

    df["Day"] = df["Arrival_Date"].dt.day

    df["DayOfWeek"] = (
        df["Arrival_Date"].dt.dayofweek
    )

    df["DayOfYear"] = (
        df["Arrival_Date"].dt.dayofyear
    )

    # =====================================
    # 6. HISTORICAL PRICE FEATURES
    # =====================================

    grouped = df.groupby(
        group_columns,
        sort=False
    )["Modal_Price"]

    # Previous available observation
    df["Lag_1"] = grouped.shift(1)

    # Previous 7 observations average
    df["Rolling_7_Mean"] = (
        df.groupby(group_columns)["Modal_Price"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=7,
                min_periods=1
            ).mean()
        )
    )

    # Previous 30 observations average
    df["Rolling_30_Mean"] = (
        df.groupby(group_columns)["Modal_Price"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=30,
                min_periods=1
            ).mean()
        )
    )

    # =====================================
    # 7. HANDLE FIRST RECORDS
    # =====================================

    # Keep first records but mark unavailable
    # historical features as missing.
    # Do not replace them with current prices.

    df["Lag_1"] = df["Lag_1"].fillna(
        pd.NA
    )

    # =====================================
    # 8. SAVE FEATURE DATASET
    # =====================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # =====================================
    # 9. DISPLAY RESULTS
    # =====================================

    print("\nFeature Engineering Completed!")

    print("Total records:", len(df))

    print("\nFeatures created:")

    print([
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "DayOfYear",
        "Lag_1",
        "Rolling_7_Mean",
        "Rolling_30_Mean"
    ])

    print("\nMissing historical values:")

    print(
        df[
            ["Lag_1", "Rolling_7_Mean",
             "Rolling_30_Mean"]
        ].isnull().sum()
    )

    print("\nSaved file:", OUTPUT_FILE)


# =====================================
# 10. MAIN
# =====================================

if __name__ == "__main__":
    create_features()
