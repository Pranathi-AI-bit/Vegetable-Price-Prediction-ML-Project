

# Python code	Purpose
# drop_duplicates()	Duplicate rows remove pannum
# to_datetime()	Date format convert pannum
# to_numeric()	Values numeric format-ku convert pannum
# dropna()	Essential missing data rows remove pannum
# sort_values()	Data sort pannum
# reset_index()	Index reset pannum
# to_csv()	Cleaned data save pannum


import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "raw" / "agmarknet_official_prices.csv"
OUTPUT_FILE = BASE_DIR / "data" / "cleaned_vegetable_prices.csv"


def preprocess_data():
    print("Loading market price data...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run python fetch_live_prices.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Support alternate date column names
    if "arrival_date" not in df.columns and "date" in df.columns:
        df.rename(columns={"date": "arrival_date"}, inplace=True)

    required_columns = [
        "state", "district", "market", "commodity",
        "arrival_date", "min_price", "max_price", "modal_price"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Remove duplicate records
    df.drop_duplicates(inplace=True)

    # Clean text columns
    text_columns = [
        "state", "district", "market", "commodity",
        "variety", "grade"
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    # Convert date
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date"], errors="coerce", dayfirst=True
    )

    # Convert price columns to numeric
    price_columns = ["min_price", "max_price", "modal_price"]

    for col in price_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove invalid records
    df.dropna(
        subset=["arrival_date", "min_price", "max_price", "modal_price"],
        inplace=True
    )

    df = df[
        (df["min_price"] >= 0) &
        (df["max_price"] >= 0) &
        (df["modal_price"] >= 0) &
        (df["min_price"] <= df["max_price"])
    ]

    # Convert rupees per quintal to rupees per kg
    df["min_price_per_kg"] = df["min_price"] / 100
    df["max_price_per_kg"] = df["max_price"] / 100
    df["modal_price_per_kg"] = df["modal_price"] / 100

    # Keep Tamil Nadu records
    df = df[
        df["state"].str.lower().str.contains("tamil nadu", na=False)
    ]

    # Standardize output column names
    df.rename(columns={
        "state": "State",
        "district": "District",
        "market": "Market",
        "commodity": "Commodity",
        "variety": "Variety",
        "grade": "Grade",
        "arrival_date": "Arrival_Date",
        "min_price": "Min_Price",
        "max_price": "Max_Price",
        "modal_price": "Modal_Price",
        "min_price_per_kg": "Min_Price_Per_Kg",
        "max_price_per_kg": "Max_Price_Per_Kg",
        "modal_price_per_kg": "Modal_Price_Per_Kg"
    }, inplace=True)

    # Sort records
    df.sort_values(
        by=["Commodity", "Market", "Arrival_Date"],
        inplace=True
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessing completed!")
    print(f"Total cleaned records: {len(df)}")
    print(f"Saved file: {OUTPUT_FILE}")
    print("\nVegetables available:")
    print(df["Commodity"].unique())


if __name__ == "__main__":
    preprocess_data()
