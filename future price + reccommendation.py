import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing

# Import data
historical_df = pd.read_csv('historical_property_data.csv')
houses = pd.read_csv('tamil_nadu.csv')

# Function to train Prophet models for each property
def train_prophet_models(historical_df):
    models = {}
    for property_id in historical_df['property_id'].unique():
        property_data = historical_df[historical_df['property_id'] == property_id][['ds', 'y']]
        model = Prophet()
        model.fit(property_data)
        models[property_id] = model
    return models

# Train Prophet models for the properties
models = train_prophet_models(historical_df)

# Save the trained Prophet models
with open('prophet_models1.pkl', 'wb') as model_file:
    pickle.dump(models, model_file)

print("Prophet models saved to 'prophet_models1.pkl'.")

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

# Load the trained Prophet models
with open('prophet_models1.pkl', 'rb') as model_file:
    loaded_models = pickle.load(model_file)

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
            print(f"Predicting future prices for property ID: {property_id}")
            model = loaded_models[property_id]
            future = model.make_future_dataframe(periods=52)  
            forecast = model.predict(future)

            # Get the historical price for 2024 and forecast for the first 5 months of 2025
            historical_price_2024 = forecast[forecast['ds'].dt.year == 2024]  
            forecast_2025 = forecast[(forecast['ds'].dt.year == 2025) & 
                                     (forecast['ds'].dt.month <= 5)][['ds', 'yhat']]  
            
            # Create a dictionary to store the historical and forecasted prices
            future_price_info = {
                'property_id': property_id,
                'historical_price_2024': historical_price_2024,  
                'forecast_2025': forecast_2025['yhat'].values.tolist()  
            }

            future_predictions.append(future_price_info)
        else:
            print(f"No model found for property ID: {property_id}")

    # Convert future_predictions to DataFrame
    predictions_df = pd.DataFrame(future_predictions)
    return predictions_df

# Function to plot 
def plot_property_prices(future_predictions):
    for idx, row in future_predictions.iterrows():
        property_id = row['property_id']
        historical_price_2024 = row['historical_price_2024']
        forecast_2025 = row['forecast_2025']

        plt.figure(figsize=(10, 5))

        # Plot historical prices for 2024
        plt.plot(historical_price_2024['ds'], historical_price_2024['yhat'], label='Historical Price 2024', color='blue', linewidth=2)

        # Plot predictions for the first 5 months of 2025 as a line
        plt.plot(pd.date_range(start='2025-01-01', periods=len(forecast_2025)), forecast_2025, label='2025 Prediction', color='green', linewidth=2)

        plt.title(f"Property Price Forecast for Property ID: {property_id}")
        plt.xlabel('Date')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

# Function to recommend properties and predict future prices
def recommend_with_future_prices(id_array):
    recommended_properties = recommend_top10_2_1_2(id_array)
    print(f"Recommended properties shape: {recommended_properties.shape}")
    
    future_predictions = predict_future_prices(recommended_properties)
    print(f"Future predictions shape: {future_predictions.shape}")

    # Display the table with the recommended properties and future prices
    print("\nRecommended Properties with Future Prices:\n", future_predictions)

    # Plot the predicted prices
    plot_property_prices(future_predictions)

# Save the entire recommendation system including models and data
def save_recommendation_system():
    recommendation_system = {
        'prophet_models': models,
        'houses_data': houses_a,
        'historical_df': historical_df
    }
    with open('recommendation_system_with_prophet1.pkl', 'wb') as model_file:
        pickle.dump(recommendation_system, model_file)

    print("Recommendation system saved to 'recommendation_system_with_prophet1.pkl'.")

# Example usage
recommended_properties_with_prices = recommend_with_future_prices([10, 1])
save_recommendation_system()

