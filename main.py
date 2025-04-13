import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import json
import os

# Configuration
API_KEY = os.getenv("api_key_data.gov.in")  # Register on data.gov.in to get API key
BASE_URL = "https://api.data.gov.in/resource/variety-wise-daily-market-prices-data-commodity"  # You'll need the exact endpoint

# Function to fetch data from the API
def fetch_crop_data(resource_id, crop_name, region, start_date, end_date):
    """
    Fetch historical crop price data from the Open Government Data Portal
    
    Parameters:
    resource_id (str): API resource ID
    crop_name (str): Name of the crop
    region (str): Region/market name
    start_date (str): Start date in YYYY-MM-DD format
    end_date (str): End date in YYYY-MM-DD format
    
    Returns:
    pandas.DataFrame: Historical price data
    """
    params = {
        "api-key": API_KEY,
        "format": "json",
        "filters[crop]": crop_name,
        "filters[region]": region,
        "filters[date_gte]": start_date,
        "filters[date_lte]": end_date
    }
    
    response = requests.get(f"{BASE_URL}{resource_id}", params=params)
    
    if response.status_code == 200:
        data = response.json()
        # Convert the JSON response to a DataFrame
        # This structure will depend on the actual API response format
        records = data.get("records", [])
        df = pd.DataFrame(records)
        # Convert date strings to datetime objects
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        # Convert price columns to numeric
        if "price" in df.columns:
            df["price"] = pd.to_numeric(df["price"])
        return df
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return pd.DataFrame()

# Function to visualize historical price data
def visualize_price_history(df, crop_name, region):
    """
    Create a visualization of historical price data
    
    Parameters:
    df (pandas.DataFrame): Historical price data
    crop_name (str): Name of the crop
    region (str): Region/market name
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["price"], marker='o', linestyle='-')
    plt.title(f"{crop_name} Price History in {region}")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# Function to build and train the prediction model
def build_prediction_model(df):
    """
    Build and train a model to predict future crop prices
    
    Parameters:
    df (pandas.DataFrame): Historical price data
    
    Returns:
    tuple: Trained model and feature engineering function
    """
    # Feature engineering - create features that might influence prices
    df = df.copy()
    
    # Add time-based features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    
    # Add lag features (previous prices)
    df["price_lag_7"] = df["price"].shift(7)
    df["price_lag_14"] = df["price"].shift(14)
    df["price_lag_30"] = df["price"].shift(30)
    
    # Add rolling statistics
    df["price_rolling_mean_7"] = df["price"].rolling(window=7).mean()
    df["price_rolling_std_7"] = df["price"].rolling(window=7).std()
    
    # Drop rows with NaN values (from lag/rolling features)
    df = df.dropna()
    
    # Select features and target
    features = ["month", "year", "day_of_year", "price_lag_7", 
                "price_lag_14", "price_lag_30", "price_rolling_mean_7", 
                "price_rolling_std_7"]
    X = df[features]
    y = df["price"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"Model Performance - MSE: {mse:.2f}, R²: {r2:.2f}")
    
    # Define a function to prepare new data for prediction
    def prepare_features(new_data):
        # Apply the same feature engineering to new data
        new_data["month"] = new_data["date"].dt.month
        new_data["year"] = new_data["date"].dt.year
        new_data["day_of_year"] = new_data["date"].dt.dayofyear
        
        # For real-time prediction, we'd use the most recent values for lag features
        latest_price = new_data["price"].iloc[-1] if not new_data.empty else 0
        new_data["price_lag_7"] = latest_price
        new_data["price_lag_14"] = latest_price
        new_data["price_lag_30"] = latest_price
        
        # For rolling stats, use the most recent values or calculate if we have enough data
        if len(new_data) >= 7:
            new_data["price_rolling_mean_7"] = new_data["price"].rolling(window=7).mean().iloc[-1]
            new_data["price_rolling_std_7"] = new_data["price"].rolling(window=7).std().iloc[-1]
        else:
            new_data["price_rolling_mean_7"] = new_data["price"].mean() if not new_data.empty else 0
            new_data["price_rolling_std_7"] = new_data["price"].std() if not new_data.empty else 0
            
        return new_data[features]
    
    return model, prepare_features

# Function to predict future prices
def predict_future_prices(model, prepare_features, historical_data, days_to_predict=30):
    """
    Predict future crop prices
    
    Parameters:
    model: Trained prediction model
    prepare_features: Function to prepare features for prediction
    historical_data (pandas.DataFrame): Historical price data
    days_to_predict (int): Number of days to predict into the future
    
    Returns:
    pandas.DataFrame: DataFrame with predicted prices
    """
    last_date = historical_data["date"].max()
    
    # Create DataFrame for future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=days_to_predict)
    future_df = pd.DataFrame({"date": future_dates})
    
    # Initialize with the last known price
    last_price = historical_data["price"].iloc[-1]
    future_df["price"] = last_price
    
    # Predict prices one day at a time (to update lag features)
    for i in range(days_to_predict):
        # Prepare features for the current prediction
        prediction_df = prepare_features(future_df.iloc[:i+1].copy())
        
        # Make prediction for the current day
        predicted_price = model.predict(prediction_df.iloc[[-1]])[0]
        
        # Update the price for the current day
        future_df.loc[i, "price"] = predicted_price
    
    return future_df

# Main Streamlit app
def main():
    st.title("Crop Price Analysis and Prediction")
    
    # Sidebar for input parameters
    st.sidebar.header("Data Selection")
    
    # Mock data for demonstration - in a real app, fetch from API
    crop_list = ["Rice", "Wheat", "Potato", "Onion", "Tomato"]
    region_list = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"]
    
    selected_crop = st.sidebar.selectbox("Select Crop", crop_list)
    selected_region = st.sidebar.selectbox("Select Region", region_list)
    
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
    
    # Resource ID for the API (this is an example, you'll need the correct one)
    resource_id = "sample_resource_id"
    
    if st.sidebar.button("Fetch Data"):
        # For demonstration, we'll use mock data since we don't have the actual API
        # In a real application, use:
        # df = fetch_crop_data(resource_id, selected_crop, selected_region, start_date, end_date)
        
        # Generate mock data for demonstration
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(date_range)
        
        # Create a price trend with some seasonality and randomness
        base_price = 50 if selected_crop in ["Rice", "Wheat"] else 20
        seasonal = 10 * np.sin(np.linspace(0, 6*np.pi, n))  # Seasonal component
        trend = np.linspace(0, 15, n)  # Upward trend
        random = np.random.normal(0, 5, n)  # Random noise
        
        prices = base_price + seasonal + trend + random
        prices = np.maximum(prices, 5)  # Ensure no negative prices
        
        df = pd.DataFrame({
            "date": date_range,
            "price": prices,
            "crop": selected_crop,
            "region": selected_region
        })
        
        # Display the data
        st.subheader("Historical Price Data")
        st.dataframe(df.head())
        
        # Visualize historical prices
        st.subheader("Price History Visualization")
        fig = visualize_price_history(df, selected_crop, selected_region)
        st.pyplot(fig)
        
        # Build prediction model
        st.subheader("Price Prediction Model")
        model, prepare_features = build_prediction_model(df)
        
        # Predict future prices
        days_to_predict = st.slider("Days to Predict", min_value=7, max_value=90, value=30)
        future_df = predict_future_prices(model, prepare_features, df, days_to_predict)
        
        # Display prediction results
        st.subheader("Price Prediction Results")
        
        # Determine if prices are predicted to rise or fall
        last_known_price = df["price"].iloc[-1]
        predicted_avg = future_df["price"].mean()
        predicted_last = future_df["price"].iloc[-1]
        
        if predicted_last > last_known_price:
            prediction_text = f"Prices are predicted to RISE by {((predicted_last/last_known_price)-1)*100:.2f}%"
            prediction_color = "green"
        else:
            prediction_text = f"Prices are predicted to FALL by {((1-(predicted_last/last_known_price)))*100:.2f}%"
            prediction_color = "red"
        
        st.markdown(f"<h3 style='color:{prediction_color}'>{prediction_text}</h3>", unsafe_allow_html=True)
        
        # Visualize prediction
        plt.figure(figsize=(12, 6))
        plt.plot(df["date"], df["price"], marker='o', linestyle='-', label='Historical')
        plt.plot(future_df["date"], future_df["price"], marker='o', linestyle='--', color='red', label='Predicted')
        plt.axhline(y=last_known_price, color='green', linestyle='-', alpha=0.3, label='Last Known Price')
        plt.title(f"{selected_crop} Price Prediction in {selected_region}")
        plt.xlabel("Date")
        plt.ylabel("Price (INR)")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)
        
        # Display tabular prediction data
        st.subheader("Predicted Prices")
        st.dataframe(future_df)

if __name__ == "__main__":
    main()
