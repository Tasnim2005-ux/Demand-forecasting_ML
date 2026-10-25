import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Load historical data and trained Linear Regression model
df = pd.read_csv("data/cleaned_sales.csv", parse_dates=["date"])
model = joblib.load("models/linear_regression.joblib")

history = df.sort_values("date")["sales"].astype(float).tolist()
last_date = df["date"].max()

def make_features(date, values):
    dow = date.dayofweek
    month = date.month

    return {
        "year": date.year,
        "month": month,
        "day": date.day,
        "day_of_week": dow,
        "day_of_year": date.dayofyear,
        "is_weekend": int(dow >= 5),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "weekday_sin": np.sin(2 * np.pi * dow / 7),
        "weekday_cos": np.cos(2 * np.pi * dow / 7),
        "sales_lag_1": values[-1],
        "sales_lag_7": values[-7],
        "sales_lag_14": values[-14],
        "sales_rolling_mean_7": np.mean(values[-7:]),
        "sales_rolling_mean_30": np.mean(values[-30:])
    }

# Recursively forecast the next 30 days
forecast_rows = []

for step in range(1, 31):
    future_date = last_date + pd.Timedelta(days=step)
    features = make_features(future_date, history)

    X_future = pd.DataFrame([features])
    prediction = float(model.predict(X_future)[0])

    forecast_rows.append({
        "date": future_date,
        "predicted_sales": prediction
    })

    # Use prediction as history for the following forecast day
    history.append(prediction)

forecast = pd.DataFrame(forecast_rows)

Path("data").mkdir(exist_ok=True)
forecast.to_csv("data/future_30_days_forecast.csv", index=False)

print("=" * 55)
print("NEXT 30 DAYS — DEMAND FORECAST")
print("=" * 55)
print("Historical data ends:", last_date.date())
print("Forecast period:", forecast["date"].min().date(),
      "to", forecast["date"].max().date())
print("Forecast days:", len(forecast))
print("Mean predicted sales:", round(forecast["predicted_sales"].mean(), 2))
print("Minimum prediction:", round(forecast["predicted_sales"].min(), 2))
print("Maximum prediction:", round(forecast["predicted_sales"].max(), 2))
print("\nFirst 10 forecast rows:")
print(forecast.head(10).assign(predicted_sales=lambda x: x["predicted_sales"].round(2)).to_string(index=False))

# Plot historical sales and future forecast
plt.figure(figsize=(14, 6))
plt.plot(df["date"].tail(90), df["sales"].tail(90),
         label="Historical Sales", linewidth=1.3)
plt.plot(forecast["date"], forecast["predicted_sales"],
         label="30-Day Forecast", marker="o", markersize=3)
plt.axvline(last_date, linestyle="--", alpha=0.7,
            label="Forecast Start")
plt.title("Daily Sales: Historical Data and 30-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/future_30_days_forecast.png", dpi=300)
plt.show()

print("\nSaved:")
print("- data/future_30_days_forecast.csv")
print("- data/future_30_days_forecast.png")

