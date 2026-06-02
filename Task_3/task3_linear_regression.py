import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------------
# Create Output Folder
# ----------------------------------

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("TASK 3 : LINEAR REGRESSION")
print("=" * 60)

# ----------------------------------
# Load Dataset
# ----------------------------------

housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["Target"] = housing.target

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ----------------------------------
# Features and Target
# ----------------------------------

X = df.drop("Target", axis=1)
y = df["Target"]

# ----------------------------------
# Train Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# ----------------------------------
# Train Model
# ----------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ----------------------------------
# Predictions
# ----------------------------------

y_pred = model.predict(X_test)

# ----------------------------------
# Evaluation Metrics
# ----------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")

print("MAE :", round(mae, 4))
print("MSE :", round(mse, 4))
print("R2  :", round(r2, 4))

# Save metrics

with open(
    "outputs/model_metrics.txt",
    "w"
) as f:

    f.write(f"MAE : {mae}\n")
    f.write(f"MSE : {mse}\n")
    f.write(f"R2 Score : {r2}\n")

# ----------------------------------
# Actual vs Predicted Plot
# ----------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")

plt.title("Actual vs Predicted House Prices")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.savefig(
    "outputs/actual_vs_predicted.png",
    bbox_inches="tight"
)

plt.close()

# ----------------------------------
# Residual Plot
# ----------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.savefig(
    "outputs/residual_plot.png",
    bbox_inches="tight"
)

plt.close()

# ----------------------------------
# Feature Importance
# ----------------------------------

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print("\nFeature Coefficients:")
print(coefficients)

coefficients.to_csv(
    "outputs/feature_coefficients.csv",
    index=False
)

# ----------------------------------
# Interpretation
# ----------------------------------

print("\nTop Positive Features:")

print(
    coefficients.head()
)

print("\nTop Negative Features:")

print(
    coefficients.tail()
)

print("\nTask Completed Successfully!")

print("\nGenerated Files:")

print("""
1. actual_vs_predicted.png
2. residual_plot.png
3. feature_coefficients.csv
4. model_metrics.txt
""")