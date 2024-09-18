import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor

# Load the dataset
data = pd.read_csv('Data_Set.csv')

# Define features and target
features = ['bedrooms', 'halls', 'size', 'location', 'bathrooms', 'balconies', 'parking_spaces', 
             'age_of_property', 'furnishing', 'facing_direction', 'floor_number', 'total_floors', 
             'has_lift', 'property_type']
target = 'price'

X = data[features]
y = data[target]

# Define preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedrooms', 'halls', 'size', 'bathrooms', 'balconies', 
                                    'parking_spaces', 'age_of_property', 'floor_number', 
                                    'total_floors']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['location', 'furnishing', 'facing_direction', 
                                                          'property_type'])
    ])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', MLPRegressor(max_iter=1000))
])

# Define parameter grid for RandomizedSearchCV
param_dist = {
    'model__hidden_layer_sizes': [(50, 50), (100, 100), (150, 150)],
    'model__activation': ['relu', 'tanh'],
    'model__solver': ['adam', 'lbfgs'],
    'model__alpha': [0.0001, 0.001, 0.01]
}

# Perform randomized search with cross-validation
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=50, cv=5, 
                                    scoring='r2', random_state=42)
random_search.fit(X, y)

# Get the best model
best_pipeline = random_search.best_estimator_

# Make predictions
y_pred = best_pipeline.predict(X)

# Evaluate model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)
accuracy = r2 * 100

# Output the updated results
print(f"Best Parameters: {random_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
print(f"Model Accuracy: {accuracy:.2f}%")

# Predict for a new example
new_example = {'bedrooms': 2, 'halls': 1, 'size': 1500, 'location': 'Adyar', 'bathrooms': 1,
                'balconies': 0, 'parking_spaces': 1, 'age_of_property': 5, 'furnishing': 'Semi-furnished',
                'facing_direction': 'North', 'floor_number': 2, 'total_floors': 14, 'has_lift': False,
                'property_type': 'Independent House'}

# Create a DataFrame for the new example
new_example_df = pd.DataFrame([new_example])

# Predict the price for the new example
predicted_price = best_pipeline.predict(new_example_df)
print(f"Predicted Price for {new_example}: {predicted_price[0]}")