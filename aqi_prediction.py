# AQI PREDICTION USING MACHINE LEARNING
# Regression + Classification | Spyder single-file project
# Six models total: 3 regression + 3 classification
# Install in Spyder console: %pip install pandas numpy matplotlib seaborn scikit-learn joblib
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report)
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
# 1. PATHS AND OUTPUT FOLDERS
DATA_PATH = r"C:\Users\aryak\Downloads\city_day (1).csv"
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
GRAPHS = os.path.join(OUT, "graphs")
MODELS = os.path.join(BASE, "models")
for folder in (OUT, GRAPHS, MODELS):
    os.makedirs(folder, exist_ok=True)
# 2. LOAD AND INSPECT DATA
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())
print("\\nMissing values:\\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "City"])
# 3. FEATURE ENGINEERING
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
pollutants = [c for c in pollutants if c in df.columns]
for col in pollutants:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["Year"] = df.Date.dt.year
df["Month"] = df.Date.dt.month
df["Day"] = df.Date.dt.day
df["DayOfWeek"] = df.Date.dt.dayofweek
df["MonthSin"] = np.sin(2 * np.pi * df.Month / 12)
df["MonthCos"] = np.cos(2 * np.pi * df.Month / 12)
numeric = pollutants + ["Year", "Month", "Day", "DayOfWeek",
                        "MonthSin", "MonthCos"]
categorical = ["City"]
features = numeric + categorical
# 4. EXPLORATORY DATA ANALYSIS
print("\\nAQI summary:\\n", df["AQI"].describe())
print("\\nAQI category counts:\\n", df["AQI_Bucket"].value_counts())
plt.figure(figsize=(8, 4))
sns.histplot(df["AQI"].dropna(), bins=40, kde=True)
plt.title("AQI Distribution")
plt.savefig(os.path.join(GRAPHS, "aqi_distribution.png"), bbox_inches="tight")
plt.show()
plt.figure(figsize=(9, 4))
df["AQI_Bucket"].value_counts().plot(kind="bar")
plt.title("AQI Category Distribution")
plt.ylabel("Records")
plt.savefig(os.path.join(GRAPHS, "aqi_categories.png"), bbox_inches="tight")
plt.show()
plt.figure(figsize=(10, 7))
sns.heatmap(df[pollutants + ["AQI"]].corr(), cmap="coolwarm")
plt.title("Pollutant and AQI Correlation")
plt.savefig(os.path.join(GRAPHS, "correlation.png"), bbox_inches="tight")
plt.show()
# Additional visualizations
monthly = df.dropna(subset=["AQI"]).groupby(df["Date"].dt.to_period("M"))["AQI"].mean()
plt.figure(figsize=(12, 4))
monthly.plot()
plt.title("Monthly Average AQI Trend")
plt.ylabel("Average AQI")
plt.savefig(os.path.join(GRAPHS, "monthly_aqi_trend.png"), bbox_inches="tight")
plt.show()
top_cities = df.groupby("City")["AQI"].mean().nlargest(10).sort_values()
plt.figure(figsize=(9, 5))
top_cities.plot(kind="barh")
plt.title("10 Cities with Highest Average AQI")
plt.xlabel("Average AQI")
plt.savefig(os.path.join(GRAPHS, "top_10_cities_aqi.png"), bbox_inches="tight")
plt.show()
plt.figure(figsize=(11, 5))
sns.boxplot(data=df[pollutants[:8]])
plt.xticks(rotation=45)
plt.title("Pollutant Distribution and Outliers")
plt.savefig(os.path.join(GRAPHS, "pollutant_boxplots.png"), bbox_inches="tight")
plt.show()
if "PM2.5" in df.columns:
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="PM2.5", y="AQI", alpha=0.35)
    plt.title("PM2.5 vs AQI")
    plt.savefig(os.path.join(GRAPHS, "pm25_vs_aqi.png"), bbox_inches="tight")
    plt.show()
# 5. OUTLIER DETECTION (retain possible genuine pollution peaks)
outliers = {}
for col in pollutants:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers[col] = int(((df[col] < q1 - 1.5 * iqr) |
                         (df[col] > q3 + 1.5 * iqr)).sum())
print("\\nIQR outlier counts:", outliers)
# 6. PREPROCESSING: fit imputers/scaler/encoder on training data only
try:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", encoder)
        ]), categorical)
    ])
    return Pipeline([("prep", prep), ("model", model)])
def split_data(data, target):
    # Chronological split: earlier 80% train, later 20% test
    data = data.dropna(subset=[target]).sort_values("Date")
    cut = int(len(data) * 0.8)
    train, test = data.iloc[:cut], data.iloc[cut:]
    return train[features], test[features], train[target], test[target]
# 7. REGRESSION: predict numerical AQI
# AQI_Bucket is not used as a feature (prevents target leakage).
Xtr, Xte, ytr, yte = split_data(df, "AQI")
reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=18, n_jobs=-1, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, random_state=42)
}
reg_results = []
for name, model in reg_models.items():
    print("\\nTraining regression:", name)
    pipe = make_pipeline(model)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    mse = mean_squared_error(yte, pred)
    reg_results.append([name, mean_absolute_error(yte, pred),
                        mse, np.sqrt(mse), r2_score(yte, pred)])
    joblib.dump(pipe, os.path.join(MODELS, "reg_" + name.replace(" ", "_") + ".joblib"))
reg = pd.DataFrame(reg_results, columns=["Model", "MAE", "MSE", "RMSE", "R2"])
reg = reg.sort_values("RMSE")
reg.to_csv(os.path.join(OUT, "regression_results.csv"), index=False)
print("\\nREGRESSION COMPARISON:\\n", reg.to_string(index=False))
plt.figure(figsize=(8, 4))
sns.barplot(data=reg, x="RMSE", y="Model")
plt.title("Regression Comparison - Lower RMSE is Better")
plt.savefig(os.path.join(GRAPHS, "regression_comparison.png"), bbox_inches="tight")
plt.show()

# 8. CLASSIFICATION: predict AQI category
# AQI is not used as a feature (prevents target leakage).
Xtr, Xte, ytr, yte = split_data(df, "AQI_Bucket")
clf_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=18, class_weight="balanced",
        n_jobs=-1, random_state=42),
    "SVM": SVC(class_weight="balanced")
}
clf_results, predictions = [], {}
for name, model in clf_models.items():
    print("\\nTraining classification:", name)
    pipe = make_pipeline(model)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    predictions[name] = pred
    clf_results.append([
        name, accuracy_score(yte, pred),
        precision_score(yte, pred, average="weighted", zero_division=0),
        recall_score(yte, pred, average="weighted", zero_division=0),
        f1_score(yte, pred, average="macro", zero_division=0)
    ])
    joblib.dump(pipe, os.path.join(MODELS, "clf_" + name.replace(" ", "_") + ".joblib"))

clf = pd.DataFrame(clf_results, columns=[
    "Model", "Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Macro"])
clf = clf.sort_values("F1_Macro", ascending=False)
clf.to_csv(os.path.join(OUT, "classification_results.csv"), index=False)
print("\\nCLASSIFICATION COMPARISON:\\n", clf.to_string(index=False))
plt.figure(figsize=(8, 4))
sns.barplot(data=clf, x="F1_Macro", y="Model")
plt.title("Classification Comparison - Higher Macro F1 is Better")
plt.savefig(os.path.join(GRAPHS, "classification_comparison.png"),
            bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 5))
clf.set_index("Model")[["Accuracy", "Precision_Weighted", "Recall_Weighted", "F1_Macro"]].plot(kind="bar")
plt.title("Classification Metrics Comparison")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(GRAPHS, "classification_all_metrics.png"), bbox_inches="tight")
plt.show()
# 9. CONFUSION MATRIX AND CLASSIFICATION REPORT
best_clf = clf.iloc[0]["Model"]
best_pred = predictions[best_clf]
labels = sorted(pd.Series(yte).unique())
cm = confusion_matrix(yte, best_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - " + best_clf)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(GRAPHS, "confusion_matrix.png"), bbox_inches="tight")
plt.show()
report = classification_report(yte, best_pred, zero_division=0)
print("\\nBEST CLASSIFIER:", best_clf, "\\n", report)
with open(os.path.join(OUT, "classification_report.txt"), "w") as file:
    file.write(report)

# 10. FINAL SUMMARY
print("\\nBest regression model (lowest RMSE):", reg.iloc[0]["Model"])
print("Best classification model (highest Macro F1):", best_clf)
print("Results saved in:", OUT)
print("Graphs saved in:", GRAPHS)
print("Trained models saved in:", MODELS)
print("\\nPROJECT COMPLETED.")
