
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from datetime import date, timedelta


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "featured_vegetable_prices.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
CSS_PATH = BASE_DIR / "assets" / "style.css"


# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Vegetable Price Prediction",
    page_icon="🥬",
    layout="wide"
)


# ==========================================
# 3. LOAD CUSTOM CSS
# ==========================================

def load_css():

    if CSS_PATH.exists():

        with open(CSS_PATH, "r", encoding="utf-8") as file:

            st.markdown(
                f"<style>{file.read()}</style>",
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "CSS file not found. Default Streamlit styling will be used."
        )


load_css()


# ==========================================
# 4. LOAD DATASET
# ==========================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Arrival_Date",
        "Modal_Price",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
        "DayOfYear",
        "Lag_1",
        "Rolling_7_Mean",
        "Rolling_30_Mean"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce"
    )

    df["Modal_Price"] = pd.to_numeric(
        df["Modal_Price"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "State",
            "District",
            "Market",
            "Commodity",
            "Arrival_Date",
            "Modal_Price"
        ]
    )

    return df


# ==========================================
# 5. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ==========================================
# 6. LOAD APPLICATION FILES
# ==========================================

try:

    df = load_data()

    model = load_model()

except Exception as e:

    st.error(
        f"Unable to load application files: {e}"
    )

    st.stop()


# ==========================================
# 7. APPLICATION HEADER
# ==========================================

st.title("🥬 Vegetable Price Prediction")

st.write(
    "Predict vegetable market prices using "
    "Machine Learning and historical market data."
)

st.caption(
    "Powered by Python, Streamlit and Random Forest Regression"
)

st.divider()


# ==========================================
# 8. SIDEBAR
# ==========================================

st.sidebar.title("🥬 Vegetable Predictor")

st.sidebar.write(
    """
    ### About This Project

    Vegetable Price Prediction is a Machine Learning
    application that estimates vegetable market prices
    using historical market data.

    **Algorithm**
    - Random Forest Regressor

    **Features**
    - Historical market prices
    - Price prediction
    - Market selection
    - Price trend analysis

    **Price Units**
    - ₹ / Quintal
    - ₹ / Kg
    """
)

st.sidebar.divider()

st.sidebar.info(
    "Predictions are estimates and are not guaranteed market prices."
)


# ==========================================
# 9. DASHBOARD METRICS
# ==========================================

st.subheader("📊 Dashboard Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Records",
    f"{len(df):,}"
)

col2.metric(
    "Total Vegetables",
    df["Commodity"].nunique()
)

col3.metric(
    "Total Markets",
    df["Market"].nunique()
)

st.divider()


# ==========================================
# 10. USER INPUT
# ==========================================

st.subheader("🔍 Enter Prediction Details")

col1, col2 = st.columns(2)

with col1:

    vegetables = sorted(
        df["Commodity"].astype(str).unique()
    )

    selected_vegetable = st.selectbox(
        "Select Vegetable",
        vegetables
    )

    vegetable_df = df[
        df["Commodity"].astype(str)
        == selected_vegetable
    ]

    states = sorted(
        vegetable_df["State"].astype(str).unique()
    )

    selected_state = st.selectbox(
        "Select State",
        states
    )

    state_df = vegetable_df[
        vegetable_df["State"].astype(str)
        == selected_state
    ]

    districts = sorted(
        state_df["District"].astype(str).unique()
    )

    selected_district = st.selectbox(
        "Select District",
        districts
    )


with col2:

    district_df = state_df[
        state_df["District"].astype(str)
        == selected_district
    ]

    markets = sorted(
        district_df["Market"].astype(str).unique()
    )

    selected_market = st.selectbox(
        "Select Market",
        markets
    )

    market_history = district_df[
        district_df["Market"].astype(str)
        == selected_market
    ].sort_values("Arrival_Date")

    latest_date = market_history[
        "Arrival_Date"
    ].max().date()

    min_prediction_date = max(
        date.today(),
        latest_date + timedelta(days=1)
    )

    prediction_date = st.date_input(
        "Select Prediction Date",
        value=min_prediction_date,
        min_value=min_prediction_date
    )


st.divider()


# ==========================================
# 11. PRICE PREDICTION FUNCTION
# ==========================================

def predict_price():

    history = df[
        (df["State"].astype(str) == selected_state)
        &
        (df["District"].astype(str) == selected_district)
        &
        (df["Market"].astype(str) == selected_market)
        &
        (df["Commodity"].astype(str) == selected_vegetable)
        &
        (df["Arrival_Date"].dt.date < prediction_date)
    ].sort_values("Arrival_Date")

    prices = history["Modal_Price"].dropna()

    if len(prices) == 0:

        st.error(
            "No historical price data is available "
            "for this selection."
        )

        return None

    # Latest available price
    lag_1 = prices.iloc[-1]

    # Previous 7 available records
    rolling_7 = prices.tail(7).mean()

    # Previous 30 available records
    rolling_30 = prices.tail(30).mean()

    # Convert prediction date into datetime
    prediction_datetime = pd.Timestamp(
        prediction_date
    )

    # Prepare input features
    input_data = pd.DataFrame([{

        "State": selected_state,

        "District": selected_district,

        "Market": selected_market,

        "Commodity": selected_vegetable,

        "Year": prediction_datetime.year,

        "Month": prediction_datetime.month,

        "Day": prediction_datetime.day,

        "DayOfWeek": prediction_datetime.dayofweek,

        "DayOfYear": prediction_datetime.dayofyear,

        "Lag_1": lag_1,

        "Rolling_7_Mean": rolling_7,

        "Rolling_30_Mean": rolling_30

    }])

    # Predict using trained Random Forest model
    predicted_price = float(
        model.predict(input_data)[0]
    )

    if not np.isfinite(predicted_price):

        st.error(
            "Model returned an invalid prediction."
        )

        return None

    # Avoid negative prices
    predicted_price = max(
        0,
        predicted_price
    )

    # Convert quintal to kilogram
    price_per_kg = predicted_price / 100

    return (
        predicted_price,
        price_per_kg,
        lag_1,
        rolling_7,
        rolling_30
    )


# ==========================================
# 12. HISTORICAL PRICE TREND
# ==========================================

st.subheader("📈 Historical Vegetable Price Trend")

selected_chart_vegetable = st.selectbox(
    "Choose Vegetable for Chart",
    sorted(df["Commodity"].astype(str).unique()),
    key="chart_vegetable"
)

chart_df = df[
    df["Commodity"].astype(str)
    == selected_chart_vegetable
]

daily_prices = (
    chart_df.groupby("Arrival_Date")["Modal_Price"]
    .mean()
    .reset_index()
    .sort_values("Arrival_Date")
)

if not daily_prices.empty:

    st.line_chart(
        daily_prices.set_index("Arrival_Date")["Modal_Price"],
        use_container_width=True
    )

else:

    st.info("No historical price data available for this vegetable.")


st.divider()


# ==========================================
# 13. PREDICTION BUTTON
# ==========================================

if st.button(
    "🔮 Predict Vegetable Price",
    type="primary",
    use_container_width=True
):

    result = predict_price()

    if result is not None:

        (
            predicted_price,
            price_per_kg,
            lag_1,
            rolling_7,
            rolling_30
        ) = result

        st.success(
            "Vegetable Price Prediction Completed!"
        )

        st.subheader("💰 Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric(
            "Predicted Price / Quintal",
            f"₹ {predicted_price:,.2f}"
        )

        col2.metric(
            "Predicted Price / Kg",
            f"₹ {price_per_kg:,.2f}"
        )

        st.divider()

        st.subheader("📋 Prediction Details")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:

            st.write(
                f"**Vegetable:** {selected_vegetable}"
            )

            st.write(
                f"**State:** {selected_state}"
            )

            st.write(
                f"**District:** {selected_district}"
            )

        with detail_col2:

            st.write(
                f"**Market:** {selected_market}"
            )

            st.write(
                f"**Prediction Date:** {prediction_date}"
            )

        st.divider()

        st.subheader("📊 Historical Price Features")

        feature_col1, feature_col2, feature_col3 = st.columns(3)

        feature_col1.metric(
            "Previous Price",
            f"₹ {lag_1:,.2f}"
        )

        feature_col2.metric(
            "7-Record Average",
            f"₹ {rolling_7:,.2f}"
        )

        feature_col3.metric(
            "30-Record Average",
            f"₹ {rolling_30:,.2f}"
        )

        st.info(
            "These are estimated prices based on historical data. "
            "Rolling averages use the latest available market "
            "records and may not represent exactly 7 or 30 calendar days."
        )


# ==========================================
# 14. HISTORICAL DATA PREVIEW
# ==========================================

st.divider()

with st.expander("📁 View Historical Vegetable Price Data"):

    preview_columns = [
        "Arrival_Date",
        "State",
        "District",
        "Market",
        "Commodity",
        "Modal_Price"
    ]

    st.dataframe(
        df[preview_columns].sort_values(
            "Arrival_Date",
            ascending=False
        ).head(100),

        use_container_width=True
    )


# ==========================================
# 15. FOOTER
# ==========================================

st.divider()

st.caption(
    "🥬 Vegetable Price Prediction | "
    "Machine Learning Project | "
    "Built using Python and Streamlit"
)
