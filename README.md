# Airbnb Price Prediction and Insights

## Table of Contents

* [Project Overview](#project-overview)
* [Problem Statement](#problem-statement)
* [Dataset Information](#dataset-information)
* [Objectives](#objectives)
* [Approach](#approach)

  * [1. Data Exploration and Preprocessing](#1-data-exploration-and-preprocessing)
  * [2. Model Development](#2-model-development)
  * [3. Model Evaluation](#3-model-evaluation)
  * [4. Insights and Recommendations](#4-insights-and-recommendations)
* [Tools and Technologies Used](#tools-and-technologies-used)
* [How to Run the Project](#how-to-run-the-project)
* [Success Criteria](#success-criteria)
* [Final Deliverables](#final-deliverables)

---

## Project Overview

Airbnb allows property owners to rent out spaces to travelers. Pricing listings effectively is key to maximizing revenue and staying competitive. This project builds a machine learning model to **predict the price of Airbnb listings** using various features such as room type, location, host details, and amenities. The analysis provides actionable insights for hosts to make data-driven pricing decisions.

---

## Problem Statement

The main objective is to create a **regression model** that accurately predicts the price of Airbnb listings. By analyzing features such as:

* Property type
* Room type
* Number of reviews
* Location
* Amenities
* Host characteristics

The goal is to help hosts understand what drives price changes and offer pricing recommendations that enhance both host revenue and guest satisfaction.

---

## Dataset Information

* **Dataset Name:** `Airbnb_data`
* **Data Source:** Provided internally or sourced from Airbnb open data repositories.
* **Features Include:**

  * Listing ID
  * Property type
  * Room type
  * Location (latitude, longitude, neighborhood)
  * Number of reviews
  * Availability
  * Minimum nights
  * Host listings count
  * Amenities
  * Price

---

## Objectives

1. Understand and clean the dataset.
2. Perform feature engineering and transformation.
3. Train regression models to predict price.
4. Evaluate model performance using appropriate metrics.
5. Extract insights for hosts to improve pricing decisions.

---

## Approach

### 1. Data Exploration and Preprocessing

* Handled missing values using imputation.
* Identified and treated outliers using statistical methods and visualization.
* Performed EDA (Exploratory Data Analysis) using seaborn and matplotlib.
* Feature engineering included:

  * Counting number of amenities
  * Converting categorical variables using one-hot encoding
  * Binning and transforming continuous variables

### 2. Model Development

* Tried multiple regression algorithms:

  * Linear Regression
  * Ridge & Lasso Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * XGBoost Regressor
* Used train-validation-test split to evaluate performance
* Hyperparameter tuning using GridSearchCV

### 3. Model Evaluation

Evaluated models using:

* **RMSE (Root Mean Squared Error)**
* **MAE (Mean Absolute Error)**
* **R² Score (Coefficient of Determination)**

Selected the best-performing model based on validation score and generalization ability.

### 4. Insights and Recommendations

* Neighborhoods and room types significantly affect pricing.
* Listings with higher number of amenities tend to have higher prices.
* Host experience (number of listings and reviews) plays a role in pricing.

---

## Tools and Technologies Used

* **Languages:** Python
* **Libraries:**

  * `pandas`, `numpy` (Data manipulation)
  * `matplotlib`, `seaborn` (Data visualization)
  * `scikit-learn` (Modeling and evaluation)
  * `xgboost` (Advanced regression modeling)
* **IDE:** Jupyter Notebook

---

## How to Run the Project

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/airbnb-price-prediction.git
   cd airbnb-price-prediction
   ```

2. **Install Required Libraries**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

4. Open the notebook file and execute the cells in order to view the entire process from data preprocessing to final predictions.

---

## Success Criteria

This project is considered successful if:

* The model achieves acceptable performance metrics on unseen test data.
* Key drivers influencing Airbnb prices are clearly identified and explained.
* The model can make reasonably accurate predictions on new listings.
* Non-technical users (e.g., Airbnb hosts) can understand and apply the insights.

---

## Final Deliverables

* Cleaned and preprocessed dataset
* Finalized machine learning model
* Visualization charts and data insights
* Evaluation metrics for model performance
* **Presentation Video** (max 5 minutes) summarizing the complete workflow and findings

