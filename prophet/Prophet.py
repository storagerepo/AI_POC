import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from prophet import Prophet

# Load historical property data
historical_df = pd.read_csv('historical_property_data.csv')

# Fine-tuning the Prophet model parameters to improve prediction accuracy
def train_prophet_models(historical_df):
    models = {}
    for property_id in historical_df['property_id'].unique():
        property_data = historical_df[historical_df['property_id'] == property_id][['ds', 'y']]
        property_data['ds'] = pd.to_datetime(property_data['ds'])
        
        # Applying log transform to stabilize prices
        property_data['y'] = np.log(property_data['y'])
        
        # Initialize Prophet model with tuned parameters
        model = Prophet(
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        # Seasonality
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        model.fit(property_data)
        models[property_id] = model
    return models

# Uncomment if retraining the models
models = train_prophet_models(historical_df)
with open('prophet_models_tuned2.pkl', 'wb') as model_file:
    pickle.dump(models, model_file)

# Load the tuned Prophet models
with open('prophet_models_tuned2.pkl', 'rb') as model_file:
    loaded_models = pickle.load(model_file)

def predict_future_prices(property_id):
    # Extract and prepare historical data
    historical_data = historical_df[historical_df['property_id'] == property_id][['ds', 'y']]
    historical_data['ds'] = pd.to_datetime(historical_data['ds'])
    historical_data.rename(columns={'ds': 'Date', 'y': 'Price'}, inplace=True)

    if property_id in loaded_models:
        model = loaded_models[property_id]
        last_date = historical_data['Date'].max()
        future = model.make_future_dataframe(periods=365, freq='D')
        future = future[future['ds'] > last_date]  # Filter future dates

        # Predict future prices
        forecast = model.predict(future)
        forecast['yhat'] = np.exp(forecast['yhat'])  # Reverse log transform

        # Combine historical and predicted data
        combined_data = pd.concat([
            historical_data[['Date', 'Price']],
            forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Predicted Price'})
        ], ignore_index=True)
        
        return combined_data
    print(f"No model found for property ID: {property_id}")
    return None

def plot_property_prices(property_id):
    combined_data = predict_future_prices(property_id)

    if combined_data is not None:
        plt.figure(figsize=(10, 5))
        combined_data.dropna(subset=['Date'], inplace=True)  # Drop NaN dates

        # Plot prices
        plt.plot(combined_data['Date'], 
                 combined_data['Predicted Price'].fillna(combined_data['Price']), 
                 label='Prices', color='blue', linewidth=2)

        # Mark the start of predictions
        last_historical_date = combined_data['Date'][combined_data['Price'].notna()].max()
        plt.axvline(x=last_historical_date, color='red', linestyle='--', label='Prediction Start')

        # Add title and labels
        plt.title(f"Property Price Forecast for Property ID: {property_id}")
        plt.xlabel('Date')
        plt.ylabel('Price / Predicted Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Display combined data
        print(combined_data)

        plt.show()

# Example usage
property_id = 78
plot_property_prices(property_id)