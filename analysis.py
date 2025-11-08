import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------------
# 1️⃣ Data Preprocessing
# -------------------------------------
df = pd.read_csv("expense_Jan_April.csv")

print("\n--- RAW DATA LOADED ---")
print(df.head())

df["date"] = pd.to_datetime(df["date"])

# Extract month
df["month"] = df["date"].dt.month

# Group by month and category
monthly_expense = df.groupby("month")["amount"].sum().reset_index()
category_expense = df.groupby("category")["amount"].sum().reset_index()

print("Monthly Summary:\n", monthly_expense)
print("Category Summary:\n", category_expense)

# -------------------------------------
# 2️⃣ Visualization — Bar + Line chart
# -------------------------------------

fig = make_subplots(specs=[[{"secondary_y": True}]])

# Bar Chart (monthly total)
fig.add_trace(
    go.Bar(x=monthly_expense["month"], y=monthly_expense["amount"], name="Expenses (Bar)")
)

# Line Chart (monthly trend)
fig.add_trace(
    go.Scatter(x=monthly_expense["month"], y=monthly_expense["amount"], mode="lines+markers",
               name="Trend (Line)"),
    secondary_y=True
)

fig.update_layout(
    title="Monthly Expense Trend (Bar + Line)",
    xaxis_title="Month",
    yaxis_title="Amount",
    template="plotly_white"
)

fig.show()

# -------------------------------------
# 3️⃣ Category-wise Expense Chart (Pie Chart)
# -------------------------------------

fig2 = go.Figure(
    data=[go.Pie(labels=category_expense["category"], values=category_expense["amount"])]
)
fig2.update_layout(
    title="Category-wise Spending Distribution"
)
fig2.show()

# -------------------------------------
# 4️⃣ Future Expense Prediction (Regression)
# -------------------------------------

X = np.array(monthly_expense["month"]).reshape(-1, 1)
y = monthly_expense["amount"].values

model = LinearRegression()
model.fit(X, y)

next_month = np.array([[monthly_expense["month"].max() + 1]])
predicted_expense = model.predict(next_month)[0]

print(f"\n🔮 Predicted expense for month {next_month[0][0]}: ₹{predicted_expense:.2f}")

# Add predicted point to visualization
fig3 = go.Figure()

# Existing Data
fig3.add_trace(go.Scatter(x=monthly_expense["month"], y=monthly_expense["amount"],
                          mode="lines+markers", name="Actual Expense"))

# Predicted Data point
fig3.add_trace(go.Scatter(x=[next_month[0][0]], y=[predicted_expense],
                          mode="markers", marker=dict(size=12, symbol="star"),
                          name="Predicted Expense"))

fig3.update_layout(
    title="Future Monthly Expense Prediction",
    xaxis_title="Month",
    yaxis_title="Amount",
    template="plotly_white"
)

fig3.show()
