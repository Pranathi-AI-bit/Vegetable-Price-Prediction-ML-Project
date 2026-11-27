
import requests
import pandas as pd

from pathlib import Path
from datetime import datetime, timedelta


# ==================================================
# 1. API CONFIGURATION
# ==================================================

BASE_URL = "https://api.dharun.dev/v1/mandi"

STATE = "Tamil Nadu"

OUTPUT_FILE = Path(
    "data/raw/agmarknet_official_prices.csv"
)

DAYS_TO_FETCH = 30

LIMIT = 1000


# ==================================================
# 2. EIGHT VEGETABLES
# ==================================================

VEGETABLES = {
    "Tomato": ["Tomato"],
    "Onion": ["Onion"],
    "Potato": ["Potato"],
    "Brinjal": ["Brinjal"],
    "Carrot": ["Carrot"],
    "Cabbage": ["Cabbage"],
    "Ladies Finger": [
        "Bhindi",
        "Bhendi",
        "Ladies Finger",
        "Lady Finger",
        "Okra"
    ],
    "Green Chilli": [
        "Green Chilli",
        "Chilli Green"
    ]
}


# ==================================================
# 3. FETCH DATA FOR ONE VEGETABLE AND DATE
# ==================================================

def fetch_prices(commodity, date):

    params = {
        "commodity": commodity,
        "state": STATE,
        "date": date,
        "limit": LIMIT
    }

    response = requests.get(
        BASE_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    result = response.json()

    if not isinstance(result, dict):
        raise ValueError("Unexpected API response format")

    if "error" in result:
        raise RuntimeError(str(result["error"]))

    return result.get("data", [])


# ==================================================
# 4. FETCH LAST 30 DAYS FOR ALL VEGETABLES
# ==================================================

def fetch_all_vegetables():

    all_records = []

    today = datetime.now().date()

    for day_number in range(DAYS_TO_FETCH):

        requested_date = today - timedelta(
            days=day_number
        )

        date_string = requested_date.strftime(
            "%Y-%m-%d"
        )

        print("\n" + "=" * 50)
        print("Fetching date:", date_string)
        print("=" * 50)

        for project_name, api_names in VEGETABLES.items():

            print(f"\nFetching {project_name}...")

            found = False

            for api_name in api_names:

                try:

                    records = fetch_prices(
                        api_name,
                        date_string
                    )

                    if records:

                        for record in records:

                            record["commodity"] = project_name

                        all_records.extend(records)

                        print(
                            f"{len(records)} records received "
                            f"using {api_name}"
                        )

                        found = True
                        break

                except Exception as error:

                    print(
                        f"Could not fetch {api_name} "
                        f"for {date_string}: {error}"
                    )

            if not found:

                print(
                    f"No data found for {project_name} "
                    f"on {date_string}"
                )

    return all_records


# ==================================================
# 5. CLEAN API DATA
# ==================================================

def clean_data(records):

    if not records:

        raise ValueError(
            "No vegetable records received from API."
        )

    df = pd.DataFrame(records)

    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    if "arrival_date" not in df.columns and "date" in df.columns:

        df = df.rename(
            columns={"date": "arrival_date"}
        )

    required_columns = [
        "state",
        "district",
        "market",
        "arrival_date",
        "min_price",
        "max_price",
        "modal_price",
        "commodity"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing API columns: {missing_columns}. "
            f"Available columns: {df.columns.tolist()}"
        )

    for column in ["variety", "grade"]:

        if column not in df.columns:

            df[column] = "Unknown"

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date"],
        dayfirst=True,
        errors="coerce"
    )

    price_columns = [
        "min_price",
        "max_price",
        "modal_price"
    ]

    for column in price_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "arrival_date",
            "min_price",
            "max_price",
            "modal_price"
        ]
    )

    df = df[
        (df["min_price"] >= 0) &
        (df["max_price"] >= df["min_price"]) &
        (df["modal_price"] >= df["min_price"]) &
        (df["modal_price"] <= df["max_price"])
    ]

    df = df[
        df["state"].astype(str).str.strip().str.lower()
        == STATE.lower()
    ]

    df = df.rename(
        columns={
            "state": "State",
            "district": "District",
            "market": "Market",
            "commodity": "Commodity",
            "variety": "Variety",
            "grade": "Grade",
            "arrival_date": "Arrival_Date",
            "min_price": "Min_Price",
            "max_price": "Max_Price",
            "modal_price": "Modal_Price"
        }
    )

    df["Arrival_Date"] = (
        df["Arrival_Date"].dt.strftime("%Y-%m-%d")
    )

    final_columns = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade",
        "Arrival_Date",
        "Min_Price",
        "Max_Price",
        "Modal_Price"
    ]

    return df[final_columns]


# ==================================================
# 6. SAVE AND UPDATE DATASET
# ==================================================

def save_data(df):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_FILE.exists():

        old_df = pd.read_csv(OUTPUT_FILE)

        df = pd.concat(
            [old_df, df],
            ignore_index=True
        )

        print("\nExisting data found. Merging records.")

    key_columns = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade",
        "Arrival_Date"
    ]

    df = df.drop_duplicates(
        subset=key_columns,
        keep="last"
    )

    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Arrival_Date"]
    )

    df = df.sort_values(
        by=[
            "State",
            "District",
            "Market",
            "Commodity",
            "Arrival_Date"
        ]
    )

    df["Arrival_Date"] = (
        df["Arrival_Date"].dt.strftime("%Y-%m-%d")
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 50)
    print("HISTORICAL DATA SAVED!")
    print("=" * 50)

    print("File:", OUTPUT_FILE)

    print("Total records:", len(df))

    print(
        "Available dates:",
        df["Arrival_Date"].nunique()
    )

    print(
        "Oldest date:",
        df["Arrival_Date"].min()
    )

    print(
        "Latest date:",
        df["Arrival_Date"].max()
    )

    print("\nVegetable-wise record counts:")

    print(
        df.groupby("Commodity").size()
    )

    print("\nPrice unit: Rupees per quintal")


# ==================================================
# 7. MAIN FUNCTION
# ==================================================

def main():

    print("=" * 50)
    print("HISTORICAL VEGETABLE PRICE DATA FETCHER")
    print("=" * 50)

    print("State:", STATE)
    print("Days to fetch:", DAYS_TO_FETCH)

    records = fetch_all_vegetables()

    print(
        "\nTotal records received:",
        len(records)
    )

    if not records:

        print("No data available.")
        return

    cleaned_df = clean_data(records)

    print(
        "\nValid records after cleaning:",
        len(cleaned_df)
    )

    save_data(cleaned_df)

    print("\nFETCHING COMPLETED!")


# ==================================================
# 8. RUN PROGRAM
# ==================================================

if __name__ == "__main__":

    main()
