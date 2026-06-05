import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ----------------------------------
# Create Output Folder
# ----------------------------------

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("TASK 5 : DECISION TREE & RANDOM FOREST")
print("=" * 60)

# ----------------------------------
# Load Dataset
# ----------------------------------

data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

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
# Decision Tree
# ----------------------------------

dt_model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

dt_model.fit(
    X_train,
    y_train
)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

print("\nDecision Tree Accuracy:",
      round(dt_accuracy,4))

# ----------------------------------
# Visualize Decision Tree
# ----------------------------------

plt.figure(figsize=(20,10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=data.target_names,
    filled=True,
    fontsize=8
)

plt.title("Decision Tree Visualization")

plt.savefig(
    "outputs/decision_tree.png",
    bbox_inches="tight"
)

plt.close()

# ----------------------------------
# Random Forest
# ----------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("Random Forest Accuracy:",
      round(rf_accuracy,4))

# ----------------------------------
# Compare Models
# ----------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        dt_accuracy,
        rf_accuracy
    ]
})

comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# ----------------------------------
# Model Comparison Plot
# ----------------------------------

plt.figure(figsize=(6,4))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.savefig(
    "outputs/model_comparison.png",
    bbox_inches="tight"
)

plt.close()

# ----------------------------------
# Feature Importance
# ----------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# Top 10 Features

top10 = importance.head(10)

plt.figure(figsize=(10,6))

plt.barh(
    top10["Feature"],
    top10["Importance"]
)

plt.title(
    "Top 10 Feature Importances"
)

plt.gca().invert_yaxis()

plt.savefig(
    "outputs/feature_importance.png",
    bbox_inches="tight"
)

plt.close()

# ----------------------------------
# Cross Validation
# ----------------------------------

cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5
)

with open(
    "outputs/cross_validation_scores.txt",
    "w"
) as f:

    f.write(
        f"Scores: {cv_scores}\n"
    )

    f.write(
        f"Mean Accuracy: {cv_scores.mean()}\n"
    )

print(
    "\nCross Validation Mean:",
    round(cv_scores.mean(),4)
)

# ----------------------------------
# Classification Reports
# ----------------------------------

with open(
    "outputs/metrics.txt",
    "w"
) as f:

    f.write(
        "Decision Tree Report\n\n"
    )

    f.write(
        classification_report(
            y_test,
            dt_pred
        )
    )

    f.write(
        "\n\nRandom Forest Report\n\n"
    )

    f.write(
        classification_report(
            y_test,
            rf_pred
        )
    )

print("\nTop Important Features:")
print(importance.head())

print("\nGenerated Outputs:")
print("""
1. decision_tree.png
2. feature_importance.png
3. model_comparison.png
4. feature_importance.csv
5. cross_validation_scores.txt
6. metrics.txt
""")

print("\nTASK COMPLETED SUCCESSFULLY!")