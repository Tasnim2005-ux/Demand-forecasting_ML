import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lr_path = Path("data/linear_regression_predictions.csv")
rf_path = Path("data/random_forest_predictions.csv")

if not lr_path.exists() or not rf_path.exists():
    raise FileNotFoundError(
        "Prediction file missing. Run both model scripts first."
    )

lr = pd.read_csv(lr_path, parse_dates=["date"])
rf = pd.read_csv(rf_path, parse_dates=["date"])

# Confirm both models use the same test dates and actual values
if not lr["date"].equals(rf["date"]):
    raise ValueError("The model test dates do not match.")

if not np.allclose(lr["actual_sales"], rf["actual_sales"]):
    raise ValueError("Actual sales values do not match.")

actual = lr["actual_sales"]

rows = []
for name, predictions in [
    ("Linear Regression", lr["predicted_sales"]),
    ("Random Forest", rf["predicted_sales"])
]:
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(actual, predictions),
        "RMSE": np.sqrt(mean_squared_error(actual, predictions)),
        "R2": r2_score(actual, predictions)
    })

comparison = pd.DataFrame(rows)
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)
print(comparison.round(4).to_string(index=False))

Path("data").mkdir(exist_ok=True)
comparison.to_csv("data/model_comparison.csv", index=False)

# Actual vs predicted sales
plt.figure(figsize=(14, 6))
plt.plot(lr["date"], actual, label="Actual Sales", linewidth=1.2)
plt.plot(lr["date"], lr["predicted_sales"],
         label="Linear Regression", alpha=0.8, linewidth=1)
plt.plot(rf["date"], rf["predicted_sales"],
         label="Random Forest", alpha=0.8, linewidth=1)
plt.title("Actual vs Predicted Daily Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/model_comparison.png", dpi=300)
plt.close()

# Random Forest feature importance
importance_path = Path("data/random_forest_feature_importance.csv")
if importance_path.exists():
    importance = pd.read_csv(importance_path, index_col=0, header=None)
    importance.columns = ["importance"]
    importance = importance.sort_values("importance", ascending=True)

    plt.figure(figsize=(10, 7))
    importance.tail(15)["importance"].plot(kind="barh")
    plt.title("Random Forest — Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("data/feature_importance.png", dpi=300)
    plt.close()

print("\nSaved:")
print("- data/model_comparison.csv")
print("- data/model_comparison.png")
if importance_path.exists():
    print("- data/feature_importance.png")

best = comparison.sort_values("MAE").iloc[0]
print(f"\nLower-MAE model: {best['Model']} (MAE={best['MAE']:.4f})")
