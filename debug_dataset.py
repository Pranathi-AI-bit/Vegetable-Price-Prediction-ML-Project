
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

FILE = BASE_DIR / "data" / "featured_vegetable_prices.csv"

print("Checking featured dataset...\n")

if not FILE.exists():
    raise FileNotFoundError("Featured dataset not found!")

df = pd.read_csv(FILE)

print("Total records:", len(df))

print("\nAvailable columns:")
print(df.columns.tolist())

# Required training columns
required = [
    "State", "District", "Market", "Commodity",
    "Year", "Month", "Day", "DayOfWeek",
    "DayOfYear", "Lag_1",
    "Rolling_7_Mean", "Rolling_30_Mean",
    "Modal_Price", "Arrival_Date"
]

print("\nMissing columns:")
print([c for c in required if c not in df.columns])

print("\nMissing values:")
print(df.isnull().sum().to_string())

print("\nData types:")
print(df.dtypes.to_string())

# Check valid rows
available = [c for c in required if c in df.columns]

print("\nValid records after dropping missing values:")
print(len(df.dropna(subset=available)))

print("\nSample records:")
print(df.head(10).to_string(index=False))
