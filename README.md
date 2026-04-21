# 📉 Telecom Customer Churn Prediction

A complete end-to-end machine learning pipeline in R for predicting customer churn in the telecom industry. The project covers data preprocessing, exploratory data analysis, feature engineering, and model comparison using multiple classification algorithms.

---

## 📌 Overview

Customer churn is one of the most critical challenges in the telecom industry. This project builds a predictive model to identify customers who are likely to leave, enabling proactive retention strategies.

---

## 🗂️ Project Structure

```
telecom-churn-prediction/
│
├── telecom.csv              # Raw dataset
├── churn_analysis.R         # Main analysis script
└── README.md
```

---

## 🔄 Pipeline

```
Data Loading → Cleaning & Preprocessing → EDA → Feature Engineering → Modeling → Evaluation
```

### 1. Data Preprocessing
- Missing value detection and removal
- Factor encoding (`Churn`, `SeniorCitizen`)
- One-hot encoding via `caret::dummyVars`
- Min-Max scaling
- Outlier capping using IQR method

### 2. Exploratory Data Analysis (EDA)
- Structural analysis with `inspectdf`
- Distribution and correlation plots
- Target variable exploration with `explore`

### 3. Feature Engineering
- VIF (Variance Inflation Factor) analysis to remove multicollinear features
- Weight of Evidence (WoE) transformation via `scorecard`
- Information Value (IV) filtering — features with IV < 0.02 removed

### 4. Models Trained
| Model | Algorithm Type |
|---|---|
| Logistic Regression (GLM) | Linear |
| Decision Tree | Non-linear |
| Random Forest | Bagging |
| XGBoost | Boosting |

### 5. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- AUC-ROC

---

## 📦 Requirements

Install the required R packages:

```r
install.packages(c(
  "tidyverse", "caret", "scales", "dplyr", "inspectdf",
  "naniar", "correlationfunnel", "explore", "SmartEDA",
  "car", "scorecard", "tidymodels", "bonsai", "discrim",
  "rsample", "yardstick", "MLmetrics", "pROC",
  "ranger", "xgboost", "rpart"
))
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/telecom-churn-prediction.git
cd telecom-churn-prediction
```

2. Place `telecom.csv` in the project root directory.

3. Open `churn_analysis.R` in RStudio and run the script.

---

## 📊 Dataset

The dataset contains telecom customer information including:
- Demographics (gender, age, senior citizen status)
- Account information (tenure, contract type, payment method)
- Services subscribed (phone, internet, streaming, etc.)
- Monthly and total charges
- **Target variable**: `Churn` (Yes/No)

> Dataset source: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 📈 Results

Model performance is compared across all algorithms using a held-out test set (20% split), with results ranked by accuracy and AUC.

---

## 🛠️ Tech Stack

![R](https://img.shields.io/badge/R-276DC3?style=flat&logo=r&logoColor=white)
![tidyverse](https://img.shields.io/badge/tidyverse-1A162D?style=flat)
![tidymodels](https://img.shields.io/badge/tidymodels-blue?style=flat)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
