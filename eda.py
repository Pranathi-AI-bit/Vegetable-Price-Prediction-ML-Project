


#Code	Meaning
#pd.read_csv()	CSV file read pannum
#df.head()	First 5 rows
#df.shape	Rows and columns count
#df.info()	Column datatype details
#df.isnull().sum()	Missing values count
#df.duplicated().sum()	Duplicate rows count
#df.describe()	Statistical summary
#groupby()	Category-wise analysis





import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "cleaned_vegetable_prices.csv"
OUTPUT_DIR = BASE_DIR / "eda_results"

OUTPUT_DIR.mkdir(exist_ok=True)


def run_eda():

    print("Loading cleaned vegetable price data...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Cleaned dataset not found. Run python preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE)
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

    # Check required columns
    required = ["Commodity", "Market", "Arrival_Date", "Modal_Price_Per_Kg"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df.dropna(subset=required, inplace=True)

    print("\n========== DATASET OVERVIEW ==========")
    print("Total records:", len(df))
    print("Vegetables:", df["Commodity"].nunique())
    print("Markets:", df["Market"].nunique())
    print("Date range:", df["Arrival_Date"].min(), "to", df["Arrival_Date"].max())

    print("\n========== VEGETABLE PRICE SUMMARY ==========")
    summary = df.groupby("Commodity")["Modal_Price_Per_Kg"].agg(
        ["count", "mean", "min", "max", "std"]
    ).round(2)

    print(summary)

    summary.to_csv(OUTPUT_DIR / "vegetable_price_summary.csv")

    # 1. Average price by vegetable
    avg_prices = df.groupby("Commodity")["Modal_Price_Per_Kg"].mean().sort_values()

    plt.figure(figsize=(10, 6))
    avg_prices.plot(kind="bar")
    plt.title("Average Price of Vegetables")
    plt.xlabel("Vegetable")
    plt.ylabel("Average Modal Price (₹/kg)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_vegetable_prices.png")
    plt.close()

    # 2. Price trends over time
    daily_prices = df.groupby(
        ["Arrival_Date", "Commodity"]
    )["Modal_Price_Per_Kg"].mean().reset_index()

    plt.figure(figsize=(12, 6))

    for commodity in daily_prices["Commodity"].unique():
        vegetable_data = daily_prices[
            daily_prices["Commodity"] == commodity
        ]

        plt.plot(
            vegetable_data["Arrival_Date"],
            vegetable_data["Modal_Price_Per_Kg"],
            label=commodity
        )

    plt.title("Vegetable Price Trends Over Time")
    plt.xlabel("Arrival Date")
    plt.ylabel("Modal Price (₹/kg)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vegetable_price_trends.png")
    plt.close()

    # 3. Market-wise average prices
    market_prices = df.groupby("Market")["Modal_Price_Per_Kg"].mean().sort_values()

    plt.figure(figsize=(10, 6))
    market_prices.plot(kind="bar")
    plt.title("Average Vegetable Prices by Market")
    plt.xlabel("Market")
    plt.ylabel("Average Modal Price (₹/kg)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "market_price_comparison.png")
    plt.close()

    # 4. Price distribution
    plt.figure(figsize=(10, 6))
    df.boxplot(
        column="Modal_Price_Per_Kg",
        by="Commodity",
        rot=45
    )
    plt.title("Vegetable Price Distribution")
    plt.suptitle("")
    plt.xlabel("Vegetable")
    plt.ylabel("Modal Price (₹/kg)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_distribution.png")
    plt.close()

    print("\nEDA completed successfully!")
    print("Charts and summary saved inside:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()
