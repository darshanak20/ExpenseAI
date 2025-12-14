import streamlit as st
import pandas as pd
import plotly.express as px

from src.categorization import categorize_expenses
from src.forecasting import predict_next_month_total
from src.anomaly import detect_anomalies



st.set_page_config(page_title="ExpenseAI", layout="centered")

st.title("ExpenseAI Dashboard")

st.write(
    "Upload your expense CSV or use sample data to view category-wise spending "
    "and expected expense for next month."
)

# -------------------------
# DATA SOURCE SELECTION
# -------------------------
st.subheader("Data Source")

data_option = st.radio(
    "Choose data source:",
    ["Upload CSV", "Use sample data"]
)

df = None

if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload expense CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif data_option == "Use sample data":
    df = pd.read_csv("data/sample_expenses.csv")

# -------------------------
# MAIN LOGIC
# -------------------------
if df is not None:

    # Basic validation
    required_cols = {"amount", "date"}
    if not required_cols.issubset(df.columns):
        st.error("CSV must contain at least 'amount' and 'date' columns.")
        st.stop()

    # -------------------------
    # CATEGORIZATION
    # -------------------------
    df = categorize_expenses(df)

    st.subheader("Category-wise Expense Breakdown")

    category_summary = (
        df.groupby("category_auto")["amount"]
        .sum()
        .reset_index()
        .sort_values("amount", ascending=False)
    )

    fig = px.pie(
        category_summary,
        names="category_auto",
        values="amount",
        title="Spending by Category (₹)",
        hole=0.3
    )

    st.plotly_chart(fig, use_container_width=True)

        # -------------------------
    # MONTHLY SPENDING TREND
    # -------------------------
    st.subheader("Monthly Spending Trend")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    monthly_spending = (
        df.dropna(subset=["date"])
        .assign(month=df["date"].dt.strftime("%b %Y"))  # <-- IMPORTANT CHANGE
        .groupby("month", sort=False)["amount"]
        .sum()
        .reset_index()
    )

    if not monthly_spending.empty:
        fig_monthly = px.line(
            monthly_spending,
            x="month",
            y="amount",
            markers=True,
            title="Total Spending Per Month (₹)"
        )

        fig_monthly.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount (₹)",
            xaxis_type="category"  # <-- FORCE categorical axis
        )

        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.info("Not enough data to display monthly spending trend.")



    # -------------------------
    # FORECASTING
    # -------------------------
    st.subheader("Next Month Expense Prediction")

    try:
        forecast_result = predict_next_month_total(df, days=30)

        predicted_amount = forecast_result["predicted_amount"]
        lower = forecast_result["lower_bound"]
        upper = forecast_result["upper_bound"]

        st.markdown(
            f"""
            **Expected expense next month:** ₹ {predicted_amount:,.2f}  
            **Expected range:** ₹ {lower:,.2f} – ₹ {upper:,.2f}
            """
        )

    except Exception as e:
        st.warning(
            "Not enough data to generate prediction. "
            "Please upload more historical expense data."
        )

else:
    st.info("Please upload a CSV or select sample data to continue.")

# -------------------------
# ANOMALY DETECTION
# -------------------------
if df is not None:

    st.subheader("Unusual Expenses")

    df = detect_anomalies(df)
    anomalies_df = df[df["is_anomaly"]]

    if anomalies_df.empty:
        st.success("No unusual expenses detected.")
    else:
        st.write(
            "The following expenses are significantly higher or unusual "
            "compared to your normal spending patterns."
        )

        display_cols = ["date", "amount", "category_auto"]
        display_cols = [col for col in display_cols if col in anomalies_df.columns]

        st.dataframe(
            anomalies_df[display_cols]
            .sort_values("amount", ascending=False)
            .reset_index(drop=True),
            use_container_width=True
        )




