## AI-Powered Cryptocurrency Price Prediction Using Machine Learning

# Project Overview
This project focuses on predicting the prices of popular cryptocurrencies—Bitcoin (BTC), Ethereum (ETH), Tether (USDT), and Binance Coin (BNB)—using historical market data from **2020 to 2025**. By leveraging various machine learning regression models, the project analyzes price trends and evaluates model performance for accurate predictions.

---

# Features
- **Historical Data Collection:** Fetches 5 years of historical price data (2020–2025) for BTC, ETH, USDT, and BNB using the `yfinance` API.
- **Exploratory Data Analysis (EDA):** Visualizes trends, patterns, and correlations in cryptocurrency prices using `matplotlib` and `seaborn`.
- **Data Preprocessing:** Cleans and prepares data for machine learning models, including feature scaling.
- **Machine Learning Models:** Implements multiple regression models to predict cryptocurrency prices:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - ElasticNet
  - Support Vector Regressor (SVR)
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - K-Nearest Neighbors (KNN) Regression
  - Multi-Layer Perceptron (MLP) Regressor
- **Model Evaluation:** Compares models using key metrics:
  - Mean Squared Error (MSE)
  - R-squared
  - Mean Absolute Error (MAE)
- **Model Saving and Loading:** Saves the trained Random Forest model and scaler using `pickle` for future predictions.

---

## Technologies and Libraries
- Python 3.x
- `yfinance` – for fetching cryptocurrency data ` !pip install yfinance`-- for installing this library
- `pandas`, `numpy` – for data manipulation
- `matplotlib`, `seaborn` – for visualization
- `scikit-learn` – for machine learning models and evaluation
- `pickle` – for saving and loading trained models

---

## Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-link>
   cd <project-folder>

# Usage

Data Fetching & Preprocessing:

The script fetches historical data for BTC, ETH, USDT, and BNB from January 2020 to the current date (2025).

Drops unnecessary columns (Open, High, Low, Dividends, Stock Splits).

Performs exploratory data analysis and visualizations.

Model Training & Evaluation:

Train multiple regression models on the processed dataset.

Evaluate using MSE, R-squared, and MAE metrics.

Visualize model performance with bar charts.

Saving & Loading Models:

The best-performing Random Forest model and MinMaxScaler are saved using pickle.

Load the saved model and scaler to make predictions on new data:

## File Structure

project-folder/
│
├──bit_coin_price_prediction.ipynb
│
├── random_forest_model.pkl    
│   └─ Saved Random Forest model for future predictions
│
├── scaler.pkl                 
│   └─ Saved MinMaxScaler for data normalization
│
└── README.md                  
    └─ Project documentation

# Results

The project evaluates multiple regression models and compares their performance using key metrics.

Bar charts display the R-squared values of each model for a clear comparison.

Random Forest Regressor often provides the best balance between accuracy and generalization.

# Output of project
<img width="1407" height="796" alt="Screenshot 2025-09-14 161250" src="https://github.com/user-attachments/assets/f25a8ad4-7486-4f8d-899e-478808629f16" />


# Future Work

Include additional features such as trading volume, market sentiment, or macroeconomic indicators.

Experiment with deep learning models (LSTM, GRU) for time-series forecasting.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Deploy the model as a web application for real-time cryptocurrency price prediction.
---
#Author

Tharuni T.

GitHub: https://github.com/Tharuniiii

LinkedIn: [https://www.linkedin.com/in/tharuni-teegala/]
