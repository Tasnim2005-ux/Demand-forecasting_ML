import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load train and test data
train = pd.read_csv("data/train.csv", parse_dates=["date"])
test = pd.read_csv("data/test.csv", parse_dates=["date"])

# Define features and target
feature_cols = [c for c in train.columns if c not in ["date", "sales"]]

X_train = train[feature_cols]
y_train = train["sales"]

X_test = test[feature_cols]
y_test = test["sales"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test sales
predictions = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("=" * 50)
print("LINEAR REGRESSION — MODEL EVALUATION")
print("=" * 50)
print(f"Training records: {len(X_train)}")
print(f"Testing records:  {len(X_test)}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# Save model and predictions
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/linear_regression.joblib")

results = pd.DataFrame({
    "date": test["date"],
    "actual_sales": y_test,
    "predicted_sales": predictions,
    "error": y_test - predictions
})

results.to_csv("data/linear_regression_predictions.csv", index=False)

print("\nFirst 10 predictions:")
print(results.head(10).round(2).to_string(index=False))
print("\nSaved model: models/linear_regression.joblib")
print("Saved predictions: data/linear_regression_predictions.csv")
