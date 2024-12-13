###Prophet with manyally trained Neural Network


import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Function to load time series data
def load_data(file_path="time_series.csv"):
    df = pd.read_csv(file_path, parse_dates=["Timestamp"])
    return df

# Function to load property price data
def load_property_prices(file_path="p.csv"):
    property_prices = pd.read_csv(file_path)
    return property_prices

# Forecast Selling Rates with Neural Network
def forecast_selling_rates(df):
    # Ensure Timestamp is in datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Filter "Sold" data
    sold_data = df[df["Interaction Type"] == "Sold"]
    sold_data["Date"] = sold_data["Timestamp"].dt.date
    sales_per_day = sold_data.groupby("Date").size().reset_index(name="y")
    sales_per_day.rename(columns={"Date": "ds"}, inplace=True)
    
    # Check if data is empty
    if sales_per_day.empty:
        print("No data available after filtering. Please check the input DataFrame.")
        return None
    
    # Aggregate weekly sales for smoothing
    sales_per_day["ds"] = pd.to_datetime(sales_per_day["ds"])
    sales_per_day = sales_per_day.resample("W-Mon", on="ds").sum().reset_index()
    
    # Initialize Prophet model with adjustments
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1 
    )
    model.add_seasonality(name="daily", period=30, fourier_order=5)  
    model.fit(sales_per_day)

    # Forecast future data
    future = model.make_future_dataframe(periods=5, freq="W-MON")
    forecast = model.predict(future)
    
    # Calculate residuals
    residuals = sales_per_day['y'] - forecast['yhat'][:len(sales_per_day)]

    # Prepare data for Neural Network (use same features as Prophet)
    sales_per_day['week_of_year'] = sales_per_day['ds'].dt.isocalendar().week  # Example of feature engineering
    features = sales_per_day[['week_of_year']]  # Add more features if necessary
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train a neural network on the residuals
    nn_model = Sequential()
    nn_model.add(Dense(64, input_dim=features_scaled.shape[1], activation='relu'))
    nn_model.add(Dense(32, activation='relu'))
    nn_model.add(Dense(1))
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the neural network
    nn_model.fit(features_scaled, residuals, epochs=100, batch_size=32)

    # Predict residuals for future data
    future_features = future[['ds']].copy()
    future_features['week_of_year'] = future_features['ds'].dt.isocalendar().week
    future_features_scaled = scaler.transform(future_features[['week_of_year']])
    predicted_residuals = nn_model.predict(future_features_scaled)

    # Combine Prophet forecast and Neural Network residuals
    forecast['yhat_nn'] = forecast['yhat'] + predicted_residuals.flatten()

    # Plot improved graph
    fig = go.Figure()

    # Actual data
    fig.add_trace(go.Scatter(
        x=sales_per_day["ds"],
        y=sales_per_day["y"],
        mode="lines+markers",
        name="Actual Sales",
        hoverinfo="x+y"
    ))

    # Prophet Forecast
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Prophet Forecasted Sales",
        hoverinfo="x+y",
        line=dict(dash="dash")
    ))

    # Neural Network Forecast
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_nn"],
        mode="lines",
        name="Improved Forecasted Sales (Prophet + NN)",
        hoverinfo="x+y",
        line=dict(color="orange")
    ))

    # Confidence intervals for Prophet
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0.5, color="lightgrey"),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        line=dict(width=0.5, color="lightgrey"),
        showlegend=False,
        fill="tonexty",
        fillcolor="rgba(173,216,230,0.3)"
    ))

    # Add title and annotations
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Properties Sold",
        hovermode="x",
        template="plotly_white",
    )

    # Show the plot
    fig.show()

    return fig

# Example of how to call the function
# Replace 'df' with your actual dataframe loaded from the CSV file
df = load_data("time_series.csv")
print(forecast_selling_rates(df))



