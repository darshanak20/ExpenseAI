import pandas as pd

# Keyword → Category mapping
KEYWORD_CATEGORY_MAP = {
    "grocery": "Food",
    "restaurant": "Food",
    "cafe": "Food",
    "coffee": "Food",

    "uber": "Transport",
    "ola": "Transport",
    "bus": "Transport",
    "train": "Transport",

    "electricity": "Bills",
    "water": "Bills",
    "internet": "Bills",
    "mobile": "Bills",
    "recharge": "Bills",

    "movie": "Entertainment",
    "netflix": "Entertainment",
    "spotify": "Entertainment",

    "amazon": "Shopping",
    "flipkart": "Shopping",
    "online": "Shopping",

    "pharmacy": "Health",
    "hospital": "Health",
    "clinic": "Health",
    "medicine": "Health",

    "rent": "Housing"
}


def keyword_categorize(text: str) -> str:
    """
    Assign category based on keywords in text.
    """
    text = str(text).lower()

    for keyword, category in KEYWORD_CATEGORY_MAP.items():
        if keyword in text:
            return category

    return "Other"


def categorize_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize expenses using description and merchant fields.
    Adds a new column: category_auto
    """

    df = df.copy()

    # Ensure columns exist
    df["description"] = df.get("description", "").astype(str)
    df["merchant"] = df.get("merchant", "").astype(str)

    # First try description
    df["category_auto"] = df["description"].apply(keyword_categorize)

    # If still 'Other', try merchant
    mask = df["category_auto"] == "Other"
    df.loc[mask, "category_auto"] = df.loc[mask, "merchant"].apply(keyword_categorize)

    return df
