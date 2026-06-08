import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ----------------------------------
# Create Output Folder
# ----------------------------------

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("TASK 6 : KNN CLASSIFICATION")
print("=" * 60)

# ----------------------------------
# Load Dataset
# ----------------------------------

iris = load_iris()

# Use only first 2 features for
# Decision Boundary Visualization

X = iris.data[:, :2]

y = iris.target

print("\nDataset Shape:", X.shape)

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

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nFeatures Normalized")

# ----------------------------------
# Experiment with K Values
# ----------------------------------

k_values = range(1, 21)

accuracy_list = []

for k in k_values:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    accuracy_list.append(acc)

# Save K Results

results = pd.DataFrame({
    "K": list(k_values),
    "Accuracy": accuracy_list
})

results.to_csv(
    "outputs/k_results.csv",
    index=False
)

best_k = results.loc[
    results["Accuracy"].idxmax(),
    "K"
]

print("\nBest K:", best_k)

# ----------------------------------
# Plot Accuracy vs K
# ----------------------------------

plt.figure(figsize=(8,5))

plt.plot(
    k_values,
    accuracy_list,
    marker='o'
)

plt.title(
    "Accuracy vs K Value"
)

plt.xlabel("K")

plt.ylabel("Accuracy")

plt.grid(True)

plt.savefig(
    "outputs/k_accuracy_plot.png",
    bbox_inches="tight"
)

plt.close()

# ----------------------------------
# Final Model
# ----------------------------------

knn = KNeighborsClassifier(
    n_neighbors=int(best_k)
)

knn.fit(
    X_train,
    y_train
)

y_pred = knn.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:",
      round(accuracy,4))

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
# Save Metrics
# ----------------------------------

with open(
    "outputs/metrics.txt",
    "w"
) as f:

    f.write(
        f"Best K: {best_k}\n"
    )

    f.write(
        f"Accuracy: {accuracy}\n"
    )

# ----------------------------------
# Decision Boundary
# ----------------------------------

x_min, x_max = (
    X_train[:, 0].min() - 1,
    X_train[:, 0].max() + 1
)

y_min, y_max = (
    X_train[:, 1].min() - 1,
    X_train[:, 1].max() + 1
)

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

Z = knn.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.4
)

plt.scatter(
    X_train[:,0],
    X_train[:,1],
    c=y_train,
    edgecolors='k'
)

plt.title(
    f"KNN Decision Boundary (K={best_k})"
)

plt.savefig(
    "outputs/decision_boundary.png",
    bbox_inches="tight"
)

plt.close()

print("\nGenerated Outputs:")

print("""
1. confusion_matrix.png
2. decision_boundary.png
3. k_accuracy_plot.png
4. classification_report.txt
5. metrics.txt
6. k_results.csv
""")

print("\nTASK COMPLETED SUCCESSFULLY!")