import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

print("=" * 50)
print("TASK 1: DATA CLEANING & PREPROCESSING")
print("=" * 50)

# Load Dataset
df = pd.read_csv("titanic.csv")

# -----------------------------------
# Basic Information
# -----------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------------
# Handle Missing Values
# -----------------------------------

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

df["Fare"] = df["Fare"].fillna(
    df["Fare"].median()
)

# Drop Cabin due to many missing values
if "Cabin" in df.columns:
    df.drop("Cabin", axis=1, inplace=True)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------------
# Encoding
# -----------------------------------

encoder = LabelEncoder()

df["Sex"] = encoder.fit_transform(df["Sex"])

df["Embarked"] = encoder.fit_transform(
    df["Embarked"]
)

print("\nCategorical Features Encoded")

# -----------------------------------
# Standardization
# -----------------------------------

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(
    df[["Age", "Fare"]]
)

print("\nNumerical Features Standardized")

# -----------------------------------
# Boxplot Before Outlier Removal
# -----------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(data=df[["Age", "Fare"]])

plt.title("Before Outlier Removal")

plt.savefig(
    "outputs/boxplot_before_outlier_removal.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------------
# Outlier Removal using IQR
# -----------------------------------

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[
    (df["Fare"] >= lower_bound)
    & (df["Fare"] <= upper_bound)
]

print("\nShape Before Outlier Removal:")
print(df.shape)

print("\nShape After Outlier Removal:")
print(df_cleaned.shape)

# -----------------------------------
# Boxplot After Outlier Removal
# -----------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_cleaned[["Age", "Fare"]])

plt.title("After Outlier Removal")

plt.savefig(
    "outputs/boxplot_after_outlier_removal.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------------
# Save Cleaned Dataset
# -----------------------------------

df_cleaned.to_csv(
    "cleaned_titanic.csv",
    index=False
)

print("\nCleaned dataset saved.")

print("\nGenerated Files:")
print("1. cleaned_titanic.csv")
print("2. outputs/boxplot_before_outlier_removal.png")
print("3. outputs/boxplot_after_outlier_removal.png")

print("\nTASK COMPLETED SUCCESSFULLY!")