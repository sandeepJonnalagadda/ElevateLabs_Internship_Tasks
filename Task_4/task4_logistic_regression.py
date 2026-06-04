import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score
)

# ----------------------------------
# Create Output Folder
# ----------------------------------

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("TASK 4 : LOGISTIC REGRESSION")
print("=" * 60)

# ----------------------------------
# Load Dataset
# ----------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = pd.Series(data.target)

print("\nDataset Shape:", X.shape)
print("Target Classes:", data.target_names)

# ----------------------------------
# Train Test Split
# ----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------
# Feature Scaling
# ----------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeatures Standardized")

# ----------------------------------
# Train Logistic Regression
# ----------------------------------

model = LogisticRegression(
    max_iter=5000
)

model.fit(
    X_train_scaled,
    y_train
)

print("Model Trained Successfully!")

# ----------------------------------
# Predictions
# ----------------------------------

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(
    X_test_scaled
)[:, 1]

# ----------------------------------
# Metrics
# ----------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nAccuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("ROC-AUC  :", round(roc_auc,4))

# Save Metrics

with open(
    "outputs/metrics.txt",
    "w"
) as f:

    f.write(f"Accuracy : {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall   : {recall}\n")
    f.write(f"ROC-AUC  : {roc_auc}\n")

# ----------------------------------
# Confusion Matrix
# ----------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(
    "outputs/confusion_matrix.png",
    bbox_inches="tight"
)

plt.close()

# ----------------------------------
# Classification Report
# ----------------------------------

report = classification_report(
    y_test,
    y_pred
)

print("\nClassification Report")
print(report)

with open(
    "outputs/classification_report.txt",
    "w"
) as f:

    f.write(report)

# ----------------------------------
# ROC Curve
# ----------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(
    "outputs/roc_curve.png",
    bbox_inches="tight"
)

plt.close()

# ----------------------------------
# Threshold Tuning
# ----------------------------------

threshold_list = []

for threshold in np.arange(
    0.1,
    1.0,
    0.1
):

    pred = (
        y_prob >= threshold
    ).astype(int)

    threshold_list.append([
        threshold,
        accuracy_score(y_test, pred),
        precision_score(y_test, pred),
        recall_score(y_test, pred)
    ])

threshold_df = pd.DataFrame(
    threshold_list,
    columns=[
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall"
    ]
)

threshold_df.to_csv(
    "outputs/threshold_results.csv",
    index=False
)

print("\nThreshold Results Saved")

# ----------------------------------
# Sigmoid Function Plot
# ----------------------------------

x = np.linspace(-10, 10, 100)

sigmoid = 1 / (
    1 + np.exp(-x)
)

plt.figure(figsize=(8,5))

plt.plot(
    x,
    sigmoid
)

plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Probability")

plt.savefig(
    "outputs/sigmoid_curve.png",
    bbox_inches="tight"
)

plt.close()

print("\nSigmoid Curve Saved")

# ----------------------------------
# Final Summary
# ----------------------------------

print("\nGenerated Outputs")

print("""
1. confusion_matrix.png
2. roc_curve.png
3. sigmoid_curve.png
4. classification_report.txt
5. metrics.txt
6. threshold_results.csv
""")

print("\nTASK COMPLETED SUCCESSFULLY!")