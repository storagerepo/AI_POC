import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import streamlit as st

# Load data
historical_df = pd.read_csv('historical_property_data.csv')
houses = pd.read_csv('tamil_nadu.csv')

# Load the trained Prophet models
with open('prophet_models1.pkl', 'rb') as model_file:
    loaded_models = pickle.load(model_file)

# Process house data
houses_a = houses[['BEDS', 'BATHS', 'SQ_FT', 'FLOORS', 'LAT', 'LONG']]
house_data_c = houses_a.values.tolist()

# Process neighborhood and style as text features
neighbors = houses['NEIGHBORHOOD'].tolist()
styles = houses['STYLE'].tolist()

# One-hot encode text data
count_vectorizer_neighbors = CountVectorizer()
count_vectorizer_styles = CountVectorizer()
X = count_vectorizer_neighbors.fit_transform(neighbors).toarray()
Y = count_vectorizer_styles.fit_transform(styles).toarray()

# Combine numerical and encoded text data into one DataFrame
house_data_c = [house_data_c[i] + list(X[i]) + list(Y[i]) for i in range(len(house_data_c))]
house_df = pd.DataFrame(house_data_c)

# Normalize data
min_max_scaler = preprocessing.MinMaxScaler()
df_normalized = pd.DataFrame(min_max_scaler.fit_transform(house_df))

# Function to select normalized data based on house ID
def data_select_nor(id):
    index_number = houses[houses['ID'] == id].index[0]
    return df_normalized.iloc[index_number].tolist()

# Function to calculate cosine similarity between two vectors
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2) if norm_v1 and norm_v2 else 0

# Generate a list of similarities between target house and all others
def recommend_list_nor(target, exclude_ids):
    similarities = []
    for i in range(len(df_normalized)):
        if houses['ID'].iloc[i] not in exclude_ids:
            similarity = cosine_similarity(target, df_normalized.iloc[i].values)
            similarities.append((i, similarity))
    return similarities

# Recommend top 10 houses based on similarity
def recommend_top10_2_1_2(id_array):
    frames = []
    for id in id_array:
        target = data_select_nor(id)
        result = recommend_list_nor(target, id_array)
        result_list = pd.DataFrame(result).sort_values(by=[1], ascending=False).head(10)
        frames.append(result_list)

    if frames:
        total = pd.concat(frames).drop_duplicates(subset=[0]).sort_values(by=[1], ascending=False).head(10)
        top10_index_array = list(total[0])
        return houses.loc[top10_index_array, :].reset_index(drop=True)
    return pd.DataFrame()

# Function to predict future prices using Prophet for recommended properties
def predict_future_prices(recommended_properties):
    future_predictions = []
    
    for idx, row in recommended_properties.iterrows():
        property_id = row['ID']
        if property_id in loaded_models:
            model = loaded_models[property_id]
            future = model.make_future_dataframe(periods=52)
            forecast = model.predict(future)
            
            # Get the historical price for 2024 and forecast for 2025
            historical_price_2024 = forecast[forecast['ds'].dt.year == 2024]
            forecast_2025 = forecast[forecast['ds'].dt.year == 2025][['ds', 'yhat']]
            
            # Create a dictionary to store the historical and forecasted prices
            future_price_info = {
                'property_id': property_id,
                'historical_price_2024': historical_price_2024[['ds', 'yhat']].to_dict('records'),
                'forecast_2025': forecast_2025.to_dict('records')
            }

            future_predictions.append(future_price_info)
    return future_predictions

# Function to plot the predicted prices for 2024 and 2025 with a line graph
def plot_property_prices(future_predictions):
    for prediction in future_predictions:
        property_id = prediction['property_id']
        
        historical_price_2024 = pd.DataFrame(prediction['historical_price_2024'])
        forecast_2025 = pd.DataFrame(prediction['forecast_2025'])

        plt.figure(figsize=(10, 5))
        
        # Plot historical prices for 2024
        plt.plot(historical_price_2024['ds'], historical_price_2024['yhat'], label='Historical Price 2024', color='blue', linewidth=2)
        
        # Plot predictions for 2025 as a line
        plt.plot(forecast_2025['ds'], forecast_2025['yhat'], label='2025 Prediction', color='green', linewidth=2)
        
        plt.title(f"Property Price Forecast for Property ID: {property_id}")
        plt.xlabel('Date')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

# Function to recommend properties and predict future prices
def recommend_with_future_prices(id_array):
    recommended_properties = recommend_top10_2_1_2(id_array)
    future_predictions = predict_future_prices(recommended_properties)

    return recommended_properties, future_predictions

# Streamlit application layout
st.title("Property Price Recommendation and Prediction")

# Input for property IDs
property_ids = st.text_input("Enter Property IDs (comma separated)", "10, 1")
id_array = [int(id.strip()) for id in property_ids.split(',')]

# Button to trigger recommendations
if st.button("Recommend Properties"):
    recommended_properties, future_predictions = recommend_with_future_prices(id_array)
    
    st.subheader("Recommended Properties")
    st.write(recommended_properties)

    st.subheader("Future Price Predictions")
    if future_predictions:
        for prediction in future_predictions:
            st.write(f"Property ID: {prediction['property_id']}")
            historical_prices = pd.DataFrame(prediction['historical_price_2024'])
            forecast_prices = pd.DataFrame(prediction['forecast_2025'])
            
            # Merge historical and forecast data for better alignment
            merged_prices = pd.merge(historical_prices, forecast_prices, on='ds', how='outer', suffixes=('_2024', '_2025'))
            st.write(merged_prices)

            plot_property_prices([prediction])  # Plot for each property prediction
            
    else:
        st.write("No predictions available for the selected properties.")










