import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --------------------------
# Load dataset
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("crypto_data_updated_13_november.csv")
    return df

# --------------------------
# Train model
# --------------------------
@st.cache_resource
@st.cache_resource
def train_model(df):
    # Drop Date column (or any non-numeric columns)
    df = df.select_dtypes(include=['float64', 'int64'])
    
    X = df.drop(columns=['Close (BTC)'])
    y = df['Close (BTC)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, X_test, y_test, y_pred, mse, r2, mae


# --------------------------
# Prediction function
# --------------------------
def predict_btc_price(model, input_data):
    prediction = model.predict(input_data)
    return prediction[0]

# --------------------------
# Main Streamlit app
# --------------------------
def main():
    st.title("📈 Predict BTC Close Price")

    # Load data
    df = load_data()

    # Train model
    model, X_test, y_test, y_pred, mse, r2, mae = train_model(df)

    st.subheader("Model Performance")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")

   # Improved Actual vs Predicted plot
    st.subheader("📊 Actual vs Predicted BTC Close Price")

    plt.figure(figsize=(10,5))
    plt.plot(y_test.values, label="Actual (BTC)", color="blue", linestyle="dashed")
    plt.plot(y_pred, label="Predicted (BTC)", color="red")
    plt.xlabel("Samples")
    plt.ylabel("BTC Close Price (USD)")
    plt.title("Actual vs Predicted BTC Close Price")
    plt.legend()
    st.pyplot(plt)


    # Sidebar inputs
    st.sidebar.title("🔢 Input Features")
    usdt_close = st.sidebar.number_input("USDT Close", min_value=0.0, format="%.2f")
    usdt_volume = st.sidebar.number_input("USDT Volume", min_value=0.0, format="%.2f")
    bnb_close = st.sidebar.number_input("BNB Close", min_value=0.0, format="%.2f")
    bnb_volume = st.sidebar.number_input("BNB Volume", min_value=0.0, format="%.2f")
    eth_close = st.sidebar.number_input("ETH Close", min_value=0.0, format="%.2f")
    eth_volume = st.sidebar.number_input("ETH Volume", min_value=0.0, format="%.2f")
    btc_volume = st.sidebar.number_input("BTC Volume", min_value=0.0, format="%.2f")

    input_data = pd.DataFrame({
        "Volume (BTC)": [btc_volume],
        "Close (ETH)": [eth_close],
        "Volume (ETH)": [eth_volume],
        "Close (USDT)": [usdt_close],
        "Volume (USDT)": [usdt_volume],
        "Close (BNB)": [bnb_close],
        "Volume (BNB)": [bnb_volume]
    })

    if st.button("🚀 Predict BTC Close Price"):
        predicted_price = predict_btc_price(model, input_data)
        st.success(f"Predicted BTC Close Price: **${predicted_price:.2f}**")

if __name__ == "__main__":
    main()
