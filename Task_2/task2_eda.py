import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -----------------------------------
# Create Output Folder
# -----------------------------------

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("TASK 2 : EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 60)

# -----------------------------------
# Load Dataset
# -----------------------------------

df = pd.read_csv("titanic.csv")

# -----------------------------------
# Basic Dataset Information
# -----------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------------
# Summary Statistics
# -----------------------------------

print("\nSummary Statistics:")
print(df.describe())

# -----------------------------------
# Mean Median Std
# -----------------------------------

numeric_cols = df.select_dtypes(include=np.number)

summary = pd.DataFrame({
    "Mean": numeric_cols.mean(),
    "Median": numeric_cols.median(),
    "Std Dev": numeric_cols.std(),
    "Minimum": numeric_cols.min(),
    "Maximum": numeric_cols.max()
})

print("\nDetailed Statistics:")
print(summary)

summary.to_csv("outputs/summary_statistics.csv")

# -----------------------------------
# Histograms
# -----------------------------------

numeric_cols.hist(
    figsize=(14, 10),
    bins=20,
    edgecolor="black"
)

plt.suptitle("Histograms of Numerical Features")

plt.tight_layout()

plt.savefig(
    "outputs/all_histograms.png",
    bbox_inches="tight"
)

plt.close()

print("\nHistogram saved.")

# -----------------------------------
# Age Distribution
# -----------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Age"],
    kde=True
)

plt.title("Age Distribution")

plt.savefig(
    "outputs/age_distribution.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------------
# Fare Distribution
# -----------------------------------

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Fare"],
    kde=True
)

plt.title("Fare Distribution")

plt.savefig(
    "outputs/fare_distribution.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------------
# Boxplots
# -----------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df[["Age", "Fare"]]
)

plt.title("Boxplot for Age and Fare")

plt.savefig(
    "outputs/boxplot_age_fare.png",
    bbox_inches="tight"
)

plt.close()

print("Boxplot saved.")

# -----------------------------------
# Survival Count
# -----------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    x="Survived",
    data=df
)

plt.title("Survival Count")

plt.savefig(
    "outputs/survival_count.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------------
# Gender vs Survival
# -----------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title("Gender vs Survival")

plt.savefig(
    "outputs/gender_vs_survival.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------------
# Passenger Class vs Survival
# -----------------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df
)

plt.title("Passenger Class vs Survival")

plt.savefig(
    "outputs/class_vs_survival.png",
    bbox_inches="tight"
)

plt.close()

# -----------------------------------
# Correlation Heatmap
# -----------------------------------

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10, 6))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "outputs/correlation_heatmap.png",
    bbox_inches="tight"
)

plt.close()

print("Heatmap saved.")

# -----------------------------------
# Pairplot
# -----------------------------------

pairplot_df = df[
    ["Survived", "Age", "Fare", "Pclass"]
].dropna()

pair = sns.pairplot(
    pairplot_df,
    hue="Survived"
)

pair.savefig(
    "outputs/pairplot.png"
)

plt.close()

print("Pairplot saved.")

# -----------------------------------
# Skewness
# -----------------------------------

print("\nSkewness Values:")
print(numeric_df.skew())

# -----------------------------------
# Observations
# -----------------------------------

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print("""
1. Fare distribution is highly right-skewed.

2. Fare contains significant outliers.

3. Female passengers have higher survival rates.

4. First-class passengers survived more often.

5. Fare and Passenger Class show correlation.

6. Age has weaker correlation with survival.

7. Visualizations reveal patterns and anomalies.

8. Dataset contains missing values in Age, Cabin, and Embarked.
""")

print("\nEDA COMPLETED SUCCESSFULLY!")

print("\nGenerated Outputs:")

print("""
1. summary_statistics.csv
2. all_histograms.png
3. age_distribution.png
4. fare_distribution.png
5. boxplot_age_fare.png
6. survival_count.png
7. gender_vs_survival.png
8. class_vs_survival.png
9. correlation_heatmap.png
10. pairplot.png
""")