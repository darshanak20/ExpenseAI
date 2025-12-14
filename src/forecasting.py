import pandas as pd
from prophet import Prophet


def prepare_daily_expense_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw expense data into daily time series
    required by Prophet.
    Output columns: ds (date), y (daily total amount)
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])

    daily_series = (
        df.groupby("date")["amount"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "amount": "y"})
    )

    return daily_series


def forecast_next_days(daily_series: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    """
    Train Prophet model and forecast next N days.
    Returns full forecast dataframe.
    """
    if len(daily_series) < 14:
        raise ValueError("Not enough data for forecasting (need at least ~14 days).")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(daily_series)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return forecast


def predict_next_month_total(df: pd.DataFrame, days: int = 30) -> dict:
    """
    Predict total expense for next N days (default = 30).
    Returns predicted amount and confidence range.
    """
    daily_series = prepare_daily_expense_series(df)
    forecast = forecast_next_days(daily_series, days)

    future_forecast = forecast.tail(days)

    return {
        "predicted_amount": future_forecast["yhat"].sum(),
        "lower_bound": future_forecast["yhat_lower"].sum(),
        "upper_bound": future_forecast["yhat_upper"].sum(),
        "daily_forecast": future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    }
