import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px

# ---------- STEP 1: Load CSV ----------
df = pd.read_csv("expense_Jan_April.csv")

print("\n--- RAW DATA LOADED ---")
print(df.head())

# ---------- STEP 2: Clean Data ----------
df["date"] = pd.to_datetime(df["date"])
df["amount"] = df["amount"].astype(float)
df["month"] = df["date"].dt.month

print("\n--- CLEANED DATA ---")
print(df.head())

# ---------- STEP 3: ML MODEL (Predict next month expense) ----------
monthly_expense = df.groupby("month")["amount"].sum().reset_index()

X = monthly_expense[["month"]]     # feature
y = monthly_expense["amount"]      # target

model = LinearRegression()
model.fit(X, y)

next_month = monthly_expense["month"].max() + 1
prediction = model.predict([[next_month]])

print(f"\nPredicted expense for next month ({next_month}): ₹{prediction[0]:.2f}")

# ---------- STEP 4: Plotly Visualization ----------
fig = px.line(monthly_expense, x="month", y="amount", title="Monthly Expense Trend")
fig.show()
