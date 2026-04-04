# 🚜 Bulldozer Sale Price Prediction: Time-Series Regression

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/c/bluebook-for-bulldozers/data)


An end-to-end machine learning project utilizing **Random Forest Regression** to predict the auction sale price of heavy equipment. This project explores advanced data preprocessing, time-series feature engineering, and hyperparameter optimization.

---

## 🔍 Problem Definition
How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?

## 📊 Dataset
The data is sourced from the **Kaggle Bluebook for Bulldozers** competition. It is a time-series dataset containing historical sales of bulldozers with over 50 different attributes (e.g., model type, size, year made, etc.).

* **Train.csv:** Sales through the end of 2011 (~400,000 examples).
* **Valid.csv:** Sales from Jan 1, 2012, to April 30, 2012 (~12,000 examples).
* **Test.csv:** Sales from May 1, 2012, to November 2012 (used for final ranking).

---

## 🏗 Project Workflow

The project follows a rigorous 6-step machine learning framework:

1.  **Data Exploration (EDA):** Identifying temporal patterns and visualizing the distribution of `SalePrice`.
2.  **Feature Engineering:** * **Date Parsing:** Enriching the dataset by extracting `saleYear`, `saleMonth`, `saleDay`, `saleDayOfWeek`, and `saleDayOfYear`.
    * **Category Encoding:** Converting string data into pandas categories and subsequently into numerical codes.
3.  **Data Imputation:** * Filling missing numerical values using the **Median** (robust to outliers).
    * Handling missing categorical data by creating a dedicated "missing" category code.
4.  **Model Selection:** Evaluating the performance of the `RandomForestRegressor`.
5.  **Hyperparameter Tuning:** Using `RandomizedSearchCV` to optimize parameters like `n_estimators`, `max_depth`, and `min_samples_leaf`.
6.  **Evaluation:** Implementing a custom function for the **RMSLE** (Root Mean Squared Log Error).

---

## 🛠 Tech Stack

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn (RandomForestRegressor) |
| **Optimization** | RandomizedSearchCV |

---

## 🚀 Key Features

* **Time-Series Alignment:** Sorting data by `saledate` to prevent "look-ahead bias" during validation.
* **Scalable Preprocessing:** A custom `preprocess_data` function that transforms raw test data into the exact format required by the trained model.
* **Feature Importance:** Analysis of which bulldozer attributes (e.g., `YearMade`, `ProductSize`) most significantly impact the final auction price.
* **Optimized Performance:** Implementing `max_samples` during tuning to speed up the iteration cycle without losing model integrity.

---

## 📈 Evaluation Metric
The evaluation metric for this competition is the **RMSLE** (Root Mean Squared Log Error) between the actual and predicted auction prices. 
> The goal is to minimize this error to build a model that performs well across both low and high price points.

---

## 📁 Project Structure

```text
Bulldozer-Price-Prediction/
├── data/                                # Dataset directory (Train, Valid, Test)
├── end-to-end-bulldozer-price-regression.ipynb # Main implementation notebook
├── README.md                            # Project documentation
└── .gitignore
