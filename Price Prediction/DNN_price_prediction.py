import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer
import joblib
import numpy as np

# Load the dataset
data = pd.read_csv('Data_Set.csv')

# Define mandatory and optional features
mandatory_features = ['bedrooms', 'sqft', 'location', 'bathrooms']
optional_features = ['halls', 'balconies', 'parking_spaces', 'age_of_property', 
                     'furnishing', 'facing_direction', 'floor_number', 'total_floors', 
                     'has_lift', 'property_type']
target = 'price'

# Select available features
available_features = [feature for feature in (mandatory_features + optional_features) if feature in data.columns]

# Ensure optional features are present as NaN if not available
for feature in optional_features:
    if feature not in data.columns:
        data[feature] = np.nan  

# Now select X and y
X = data[available_features]
y = data[target]

# Define preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values in numeric columns
            ('scaler', StandardScaler())]), 
         [col for col in available_features if col not in ['location', 'furnishing', 'facing_direction', 'property_type']]),
        
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values in categorical columns
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), 
         ['location', 'furnishing', 'facing_direction', 'property_type'])
    ],
    remainder='passthrough'  # Leave other columns unchanged if they exist
)

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', MLPRegressor(max_iter=300))
])

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'model__hidden_layer_sizes': [(50,), (100,), (150,)],
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam', 'lbfgs'],
    'model__alpha': [0.0001, 0.001, 0.01]
}

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=4, cv=3,
                                    scoring='r2', random_state=42, n_jobs=-1)
random_search.fit(X, y)

# Save the best model
joblib.dump(random_search.best_estimator_, 'Price_Prediction.pkl')

# Print the best parameters and evaluation metrics
best_pipeline = random_search.best_estimator_
y_pred = best_pipeline.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
accuracy = r2 * 100

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(f"Model Accuracy: {accuracy:.2f}%")

# Load the saved model
loaded_model = joblib.load('Price_Prediction.pkl')

# Create a DataFrame for new data
new_data = pd.DataFrame({
    'bedrooms': [3],
    'sqft': [2000],
    'location': ['New York'],
    'bathrooms': [2],
    'halls': [1],
    'balconies': [1],

})

# Ensure all required features are present
for feature in optional_features:
    if feature not in new_data.columns:
        new_data[feature] = np.nan

# Preprocess and make predictions
predicted_price = loaded_model.predict(new_data[available_features])

print(f"Predicted Price: {predicted_price[0]}")