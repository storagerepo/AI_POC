# House Price Prediction using PyTorch

This project implements a deep learning model to predict house prices using PyTorch. The model utilizes both numerical and categorical features to make accurate price predictions.

## Project Structure

- `property_price_pred.py`: Main script containing the model implementation
- `house_data/`: Directory containing the dataset
  - `houseprice.csv`: Housing dataset used for training

## Features

The model automatically selects the most relevant features based on correlation with the sale price:

### Numerical Features
- Features with correlation > 0.5 with sale price are selected
- Missing values are handled using median imputation
- Features are standardized using StandardScaler

### Categorical Features
- Top categorical features with less than 20 unique values are selected
- Missing values are filled using mode imputation
- Features are encoded using LabelEncoder

## Model Architecture

The neural network model consists of:
- Multiple dense layers with ReLU activation
- Batch normalization layers
- Dropout layers for regularization
- Residual connections for better gradient flow

## Training

The model is trained with the following parameters:
- Batch Size: 32
- Learning Rate: 0.0001
- Epochs: 500
- Early Stopping Patience: 50
- Optimizer: Adam
- Loss Function: Mean Squared Error

## Output

The model generates predictions and visualizes them in `price_predictions.png`, showing:
- Actual vs Predicted prices
- Price trends over time
- Prediction confidence intervals

![Price Predictions](price_predictions.png)

## Performance Metrics

The model's performance is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R²) Score

## Hardware Acceleration

The code automatically detects and utilizes MPS (Metal Performance Shaders) if available on Mac devices, falling back to CPU if unavailable.

## Future Predictions

The model includes functionality to predict future house prices while maintaining trend stability, making it useful for long-term price forecasting.
