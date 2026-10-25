import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the same training and testing data
train = pd.read_csv("data/train.csv", parse_dates=["date"])
test = pd.read_csv("data/test.csv", parse_dates=["date"])

feature_cols = [
    c for c in train.columns
    if c not in ["date", "sales"]
]

X_train = train[feature_cols]
y_train = train["sales"]

X_test = test[feature_cols]
y_test = test["sales"]

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("=" * 50)
print("RANDOM FOREST — MODEL EVALUATION")
print("=" * 50)
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Save model and predictions
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/random_forest.joblib")

results = pd.DataFrame({
    "date": test["date"],
    "actual_sales": y_test,
    "predicted_sales": predictions,
    "error": y_test - predictions
})

results.to_csv("data/random_forest_predictions.csv", index=False)

# Show the most influential features
importance = pd.Series(
    model.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10).round(4).to_string())

importance.to_csv("data/random_forest_feature_importance.csv")

print("\nSaved model: models/random_forest.joblib")
print("Saved predictions: data/random_forest_predictions.csv")
print("Saved feature importance: data/random_forest_feature_importance.csv")
