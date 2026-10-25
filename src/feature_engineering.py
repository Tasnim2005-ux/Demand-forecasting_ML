import pandas as pd
import numpy as np
from pathlib import Path

# Load cleaned dataset
df = pd.read_csv("data/cleaned_sales.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["day_of_year"] = df["date"].dt.dayofyear
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Cyclical calendar features
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["weekday_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# Past sales features
for lag in [1, 7, 14]:
    df[f"sales_lag_{lag}"] = df["sales"].shift(lag)

# Rolling averages use only previous days
df["sales_rolling_mean_7"] = df["sales"].shift(1).rolling(7).mean()
df["sales_rolling_mean_30"] = df["sales"].shift(1).rolling(30).mean()

# Remove rows without enough historical data
df = df.dropna().reset_index(drop=True)

# Chronological 80/20 split
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

# Save outputs
Path("data").mkdir(exist_ok=True)
df.to_csv("data/model_ready.csv", index=False)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

feature_cols = [c for c in df.columns if c not in ["date", "sales"]]

print("FEATURE ENGINEERING COMPLETED!")
print("Total usable rows:", len(df))
print("Number of features:", len(feature_cols))
print("Features:", feature_cols)
print("Training rows:", len(train))
print("Testing rows:", len(test))
print("Training period:", train["date"].min().date(), "to", train["date"].max().date())
print("Testing period:", test["date"].min().date(), "to", test["date"].max().date())
print("Missing values in prepared data:", int(df.isnull().sum().sum()))
print("\nSaved files:")
print("- data/model_ready.csv")
print("- data/train.csv")
print("- data/test.csv")
