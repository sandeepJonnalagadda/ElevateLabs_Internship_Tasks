# Task 2: Exploratory Data Analysis (EDA)

## Objective

The objective of this task is to perform Exploratory Data Analysis (EDA) on the Titanic dataset to understand the data through statistical summaries and visualizations. EDA helps identify patterns, relationships, trends, anomalies, and potential issues before building Machine Learning models.

---

## Dataset

**Titanic Dataset**

The dataset contains information about Titanic passengers, including:

* Passenger Class
* Age
* Gender
* Fare
* Survival Status
* Embarkation Port

---

## Tools & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## Tasks Performed

### 1. Data Exploration

* Loaded the dataset using Pandas.
* Examined dataset shape, columns, and data types.
* Identified missing values.

### 2. Descriptive Statistics

Generated summary statistics including:

* Mean
* Median
* Standard Deviation
* Minimum Value
* Maximum Value

### 3. Data Visualization

#### Histograms

Created histograms to understand the distribution of numerical features.

#### Boxplots

Used boxplots to identify outliers and analyze data spread.

#### Count Plots

Visualized:

* Survival Distribution
* Gender vs Survival
* Passenger Class vs Survival

#### Correlation Heatmap

Analyzed relationships between numerical features using a correlation matrix.

#### Pairplot

Visualized pairwise relationships among important features.

---

## Key Insights

### Survival Analysis

* Female passengers had a higher survival rate compared to male passengers.
* First-class passengers were more likely to survive than third-class passengers.

### Fare Analysis

* Fare distribution is highly right-skewed.
* Several passengers paid significantly higher fares, resulting in outliers.

### Age Analysis

* Most passengers were between 20 and 40 years old.
* Age showed a weaker relationship with survival compared to class and gender.

### Correlation Analysis

* Passenger Class and Fare are negatively correlated.
* Survival has noticeable relationships with Fare and Passenger Class.

---

## Generated Outputs

The script automatically generates:

* summary_statistics.csv
* all_histograms.png
* age_distribution.png
* fare_distribution.png
* boxplot_age_fare.png
* survival_count.png
* gender_vs_survival.png
* class_vs_survival.png
* correlation_heatmap.png
* pairplot.png

All outputs are stored in the `outputs/` directory.

---

## Project Structure

```text
Task_2/
│
├── titanic.csv
├── task2_eda.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── summary_statistics.csv
    ├── all_histograms.png
    ├── age_distribution.png
    ├── fare_distribution.png
    ├── boxplot_age_fare.png
    ├── survival_count.png
    ├── gender_vs_survival.png
    ├── class_vs_survival.png
    ├── correlation_heatmap.png
    └── pairplot.png
```

---

## How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Script

```bash
python task2_eda.py
```

---

## Learning Outcomes

Through this task, I learned:

* Data Exploration Techniques
* Descriptive Statistics
* Data Visualization
* Correlation Analysis
* Pattern Recognition
* Outlier Detection
* Feature Relationship Analysis

---

## Author

**Sandeep Jonnalagadda**

AI & ML Internship – Task 2
