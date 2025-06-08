# Airbnb Price Prediction Using Machine Learning (Part 1)

## Project Overview

This project aims to build a machine learning model that predicts the **log-transformed price** of Airbnb listings. By analyzing various features like property type, location, host experience, and amenities, the model helps estimate listing prices accurately. The model used is a **Random Forest Regressor**, and it is evaluated using common performance metrics to ensure reliability.

---

## What This Project Does

* Loads and explores Airbnb listing data.
* Cleans and preprocesses the data, handling missing values and outliers.
* Creates new features from existing data (feature engineering), such as host experience duration and review dates.
* Converts categorical data into numerical format for model use.
* Scales numerical features for better model performance.
* Builds and trains a Random Forest regression model to predict prices.
* Evaluates the model using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.
* Visualizes the results and model predictions to understand performance and errors.

---

## Why This Is Important

Pricing an Airbnb listing correctly is critical for hosts to maximize their earnings and stay competitive. This model helps by:

* Providing data-driven price predictions based on listing details.
* Highlighting key factors that influence pricing.
* Offering insights to improve listing strategies.

---

## Step-by-Step Process

1. **Data Loading & Exploration:**
   Read the data and understand its structure, check for missing values, and get initial statistical summaries.

2. **Data Visualization:**
   Visualize the distribution of prices (after log transformation) and identify outliers.

3. **Feature Engineering:**
   Extract new features such as:

   * Year and month of first and last reviews.
   * Host experience measured in days since first listing.

4. **Categorical Encoding:**
   Convert categorical variables like property type and neighborhood into numerical format using one-hot encoding.

5. **Data Cleaning & Scaling:**

   * Clean zipcode data to numerical format.
   * Impute missing numerical values using medians.
   * Standardize features using scaling for uniformity.

6. **Model Development:**
   Train a Random Forest Regressor on the processed data with a train-test split.

7. **Preprocessing Pipeline:**
   Use pipelines to ensure consistent data transformation during training and testing phases.

8. **Model Evaluation:**
   Evaluate model accuracy using:

   * MAE (lower is better)
   * RMSE (lower is better)
   * R² (higher is better, here \~64.5%)

9. **Visualization of Predictions:**
   Plot actual vs predicted prices and analyze residuals to check for errors or biases.

---

## Results Summary

* **MAE:** 0.3087
* **RMSE:** 0.4238
* **R² Score:** 0.6449 (Model explains \~64.5% of price variance)

The model performs well in capturing the complexity of Airbnb pricing, making it useful for price estimation and decision-making.

---

## Tools and Libraries Used

* Python
* pandas, numpy (data handling)
* matplotlib, seaborn (visualization)
* scikit-learn (modeling and preprocessing)
* RandomForestRegressor (regression model)

---

## How to Use This Project

* Load the dataset (`airbnb_data.csv`).
* Follow the preprocessing and feature engineering steps to prepare data.
* Train the Random Forest model on training data.
* Evaluate the model on test data.
* Use the model to predict prices for new Airbnb listings.

---

## Conclusion

This project demonstrates how to build an effective machine learning pipeline to predict Airbnb listing prices. The model and analysis provide valuable insights to hosts and stakeholders, helping them understand key pricing factors and improve revenue strategies.
