import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.02
) -> pd.DataFrame:
    """
    Detect unusual expense amounts using Isolation Forest.

    Adds a new column:
    - is_anomaly : True / False
    """

    df = df.copy()

    # Ensure amount is numeric
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    # Fit model on amount column
    predictions = model.fit_predict(df[["amount"]])

    # Convert predictions to boolean flag
    df["is_anomaly"] = predictions == -1

    return df
