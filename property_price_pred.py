# %%
# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from pathlib import Path
import requests
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


USE_MPS = True 
if USE_MPS and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# %%
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 1000
PATIENCE = 50
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 0.01  # L2 regularization

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# %%
# Data paths and loading
data_path = Path("house_data/")
csv_path = data_path / "houseprice.csv"

if data_path.is_dir():
    print(f"{data_path} already exists")
else:
    print(f"{data_path} creating...")
    data_path.mkdir(parents=True, exist_ok=True)

if csv_path.is_file():
    print(f"{csv_path} already exists, skipping download...")
else:
    print(f"Downloading {csv_path}....")
    request = requests.get("https://raw.githubusercontent.com/krishnaik06/Pytorch-Tutorial/master/houseprice.csv")
    with open(csv_path, "wb") as file:
        file.write(request.content)

# %%
# Load and analyze data
df = pd.read_csv(csv_path)

# Feature selection based on correlation with raw (unscaled) data
correlations = df.select_dtypes(include=['int64', 'float64']).corr()['SalePrice']
numerical_features = [
    'OverallQual',   # Overall material and finish quality
    'GrLivArea',     # Above ground living area
    'GarageCars',    # Size of garage in car capacity
    'GarageArea',    # Size of garage in square feet
    'TotalBsmtSF',   # Total basement square footage
    '1stFlrSF',      # First Floor square feet
    'FullBath',      # Number of full bathrooms
    'TotRmsAbvGrd',  # Total rooms above ground (excluding bathrooms)
    'YearBuilt',     # Original construction date
    'YearRemodAdd',  # Remodel date
    'LotArea',       # Lot size in square feet
    '2ndFlrSF',      # Second floor square feet
    'BsmtFinSF1'     # Type 1 finished square feet
]

categorical_features = [
    'MSZoning',      # Identifies the general zoning classification
    'Street',        # Type of road access
    'Alley',         # Type of alley access
    'LotShape',      # General shape of property
    'LandContour',   # Flatness of the property
    'Neighborhood',  # Physical locations within Ames city limits
    'BldgType',      # Type of dwelling
    'HouseStyle',    # Style of dwelling
    'RoofStyle',     # Type of roof
    'Exterior1st'    # Exterior covering on house
]

print("\nSelected features:")
print("Numerical features with correlations:")
for feat in numerical_features:
    print(f"{feat}: {correlations[feat]:.3f}")
print("\nCategorical features:", categorical_features)

# Handle missing values
print("\nHandling missing values:")
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

# Feature engineering
print("\nEngineering new features...")

# Age features
df['HouseAge'] = datetime.now().year - df['YearBuilt']
df['LastRemodAge'] = datetime.now().year - df['YearRemodAdd']
df['RemodAge'] = df['YearRemodAdd'] - df['YearBuilt']

# Area interactions
df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
df['AvgRoomSize'] = df['GrLivArea'] / df['TotRmsAbvGrd']
df['TotalBathrooms'] = df['FullBath'] + df['HalfBath'] * 0.5

# Quality interactions
df['QualityAge'] = df['OverallQual'] * df['HouseAge']
df['QualityArea'] = df['OverallQual'] * df['TotalSF']

# Add engineered features to numerical_features
engineered_features = ['HouseAge', 'LastRemodAge', 'RemodAge', 'TotalSF', 
                      'AvgRoomSize', 'TotalBathrooms', 'QualityAge', 'QualityArea']
numerical_features.extend(engineered_features)

# Encode categorical features
label_encoders = {}
for c in categorical_features:
    label_encoders[c] = LabelEncoder()
    df[c] = label_encoders[c].fit_transform(df[c])
    print(f"\nEncoding for {c}:")
    for i, category in enumerate(label_encoders[c].classes_):
        print(f"{category} -> {i}")

# Add time feature
def add_time_feature(df):
    current_year = datetime.now().year
    df = df.copy()
    
    # Initialize YearsFromNow with a small range of values for training
    # This helps the model learn the time relationship better
    df['YearsFromNow'] = np.random.uniform(-2, 5, len(df))
    
    return df, current_year

df, current_year = add_time_feature(df)

# Prepare features and target
X = df[numerical_features + categorical_features + ['YearsFromNow']]
y = df['SalePrice']

# Scale numerical features
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Scale YearsFromNow separately to keep it in a reasonable range
years_scaler = StandardScaler()
X['YearsFromNow'] = years_scaler.fit_transform(X[['YearsFromNow']])

# Scale the target variable
y_scaler = StandardScaler()
y = pd.DataFrame(y_scaler.fit_transform(y.values.reshape(-1, 1)), columns=['SalePrice'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Ensure X_train and X_test remain as DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
y_train = pd.DataFrame(y_train, columns=['SalePrice'])
y_test = pd.DataFrame(y_test, columns=['SalePrice'])

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Save feature information
feature_info = {
    'numerical_features': numerical_features,
    'categorical_features': categorical_features,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'y_scaler': y_scaler,
    'years_scaler': years_scaler
}

# %%
# Create Dataset class
class HousePriceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values).to(device)
        self.y = torch.FloatTensor(y.values).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create datasets and dataloaders
train_dataset = HousePriceDataset(X_train, y_train)
test_dataset = HousePriceDataset(X_test, y_test)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# %%
# Create model
class HousePriceModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        # Deep layers with residual connections
        self.deep_layer1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        self.deep_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        self.deep_layer3 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.2)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size//2, 1)
        )
        
    def forward(self, x):
        # Input processing
        x = self.input_layer(x)
        
        # Residual connections
        identity1 = x
        x = self.deep_layer1(x) + identity1
        
        identity2 = x
        x = self.deep_layer2(x) + identity2
        
        identity3 = x
        x = self.deep_layer3(x) + identity3
        
        # Output processing
        x = self.output_layers(x)
        return x

# Initialize model, loss, and optimizer
model = HousePriceModel(input_size=len(X.columns)).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# %%
# Training and testing steps
def train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> float:
    model.train()
    total_loss = 0

    for batch, (X, y) in enumerate(data_loader):
        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()

    return total_loss / len(data_loader)

def test_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
) -> tuple[float, list, list]:
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            # Forward pass
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            
            # Accumulate loss
            total_loss += loss.item()
            
            predictions.extend(test_pred.cpu().numpy().flatten())
            actuals.extend(y.cpu().numpy().flatten())
    
    return total_loss / len(data_loader), predictions, actuals

# %%
# Training loop
def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int = EPOCHS,
    patience: int = PATIENCE
) -> tuple[torch.nn.Module, float]:
    best_loss = float('inf')
    patience_counter = 0

    for e in tqdm(range(epochs)):
        avg_loss = train_step(
            model=model,
            data_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if e % 10 == 0:
            print(f"Epoch [{e}/{epochs}], Loss: {avg_loss:.4f}")
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {e} epochs")
            break
    
    return model, best_loss

# %%
# Testing function
def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module
) -> dict:
    """
    Test the model and calculate metrics
    """
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.inference_mode():
        for X, y in test_loader:
            # Forward pass
            y_pred = model(X)
            
            # Calculate loss
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
            
            # Store predictions and actuals
            predictions.extend(y_pred.cpu().numpy().flatten())
            actuals.extend(y.cpu().numpy().flatten())
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': predictions,
        'actuals': actuals
    }

# %%
# Function to predict future prices with trend stability
def predict_future_price(
    model: torch.nn.Module,
    features: pd.Series,
    future_year: int,
    current_year: int
) -> float:
    # Create a copy of features to avoid modifying the original
    future_features = features.copy()
    
    # Update the time-based feature
    years_from_now = future_year - current_year
    
    # Scale YearsFromNow using the same scaler used in training
    future_features['YearsFromNow'] = years_scaler.transform([[years_from_now]])[0][0]
    
    # Convert to tensor and make prediction
    with torch.inference_mode():
        if isinstance(future_features, pd.Series):
            features_tensor = torch.FloatTensor(future_features.values).unsqueeze(0).to(device)
        else:
            features_tensor = torch.FloatTensor(future_features).unsqueeze(0).to(device)
        
        prediction = model(features_tensor)
        predicted_price = y_scaler.inverse_transform(prediction.cpu().numpy())[0][0]
    
    # Apply a simple appreciation rate based on historical average
    # Typical real estate appreciation is 3-4% per year
    base_price = predicted_price
    appreciation_rate = 1.035  # 3.5% annual appreciation
    
    if years_from_now > 0:
        predicted_price = base_price * (appreciation_rate ** years_from_now)
    
    return predicted_price

# %%
# Function to visualize price predictions
def visualize_price_predictions(
    current_year: int,
    predictions: dict,
    title: str = "House Price Predictions Over Time"
):
    # Set the style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure and axis with larger size
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    years = list(predictions.keys())
    prices = list(predictions.values())
    
    # Create the line plot
    plt.plot(years, prices, marker='o', linewidth=2, markersize=8)
    
    # Add price annotations
    for year, price in zip(years, prices):
        plt.annotate(
            f'${price:,.0f}',
            (year, price),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=9,
            bbox=dict(
                boxstyle='round,pad=0.5',
                fc='yellow',
                alpha=0.3
            )
        )
    
    # Customize the plot
    plt.title(title, fontsize=15, pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Predicted Price ($)', fontsize=12)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format y-axis with dollar signs and commas
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add a light background grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('price_predictions.png', dpi=300, bbox_inches='tight')
    
    # Clear the current figure to free memory
    plt.close()
    
    return plt.gcf()

# %%
# Function to test saved model
def test_saved_model(house_index=0):
 
    with open('house_price_model.pkl', 'rb') as f:
        loaded_artifacts = pickle.load(f)
    
    # Create a new model instance
    input_size = len(numerical_features + categorical_features + ['YearsFromNow'])
    loaded_model = HousePriceModel(input_size=input_size).to(device)
    # Load the state dict
    loaded_model.load_state_dict(loaded_artifacts['model_state_dict'])
    loaded_model.eval()

    # Extract all components
    loaded_encoders = loaded_artifacts['artifacts']['label_encoders']
    loaded_scaler = loaded_artifacts['artifacts']['scaler']
    loaded_y_scaler = loaded_artifacts['artifacts']['y_scaler']  # Extract y_scaler
    loaded_num_features = loaded_artifacts['artifacts']['numerical_features']
    loaded_cat_features = loaded_artifacts['artifacts']['categorical_features']
    loaded_current_year = datetime.now().year
    
    # Get the specified house from the test set
    try:
        sample_house = X_test.iloc[house_index:house_index+1]  # Get specified row as DataFrame
        actual_price_scaled = y_test.iloc[house_index, 0]  # Get specified value
    except IndexError:
        print(f"\nError: House index {house_index} is out of range. Test set has {len(X_test)} houses.")
        return None
    
    # Make prediction
    loaded_model.eval()
    with torch.no_grad():
        sample_tensor = torch.FloatTensor(sample_house.values).to(device)
        predicted_price_scaled = loaded_model(sample_tensor)
        predicted_price_scaled = predicted_price_scaled.cpu().numpy()[0][0]
    
    # Inverse transform the scaled values to get actual prices
    actual_price = loaded_y_scaler.inverse_transform([[actual_price_scaled]])[0][0]
    predicted_price = loaded_y_scaler.inverse_transform([[predicted_price_scaled]])[0][0]
    
    print(f"\nModel Testing Results for House #{house_index}:")
    print(f"Features of the house:")
    for feature in numerical_features:
        print(f"{feature}: {sample_house[feature].iloc[0]:,.2f}")
    for feature in categorical_features:
        print(f"{feature}: {sample_house[feature].iloc[0]}")
    print(f"\nActual Price: ${actual_price:,.2f}")
    print(f"Predicted Price: ${predicted_price:,.2f}")
    print(f"Difference: ${abs(actual_price - predicted_price):,.2f}")
    print(f"Percentage Difference: {abs(actual_price - predicted_price) / actual_price * 100:.2f}%")
    
    return loaded_model

# %%
# Function to make predictions for multiple years
def test_predictions(model, num_years=5):
  
    predictions = {}
    
    # Use the first test sample
    sample_house = X_test.iloc[0]

    for year in range(datetime.now().year + 1, datetime.now().year + num_years + 1):  
        predicted_price = predict_future_price(
            model, sample_house, year, datetime.now().year
        )
        predictions[year] = predicted_price
        print(f"Predicted price for {year}: ${predicted_price:,.2f}")

    # Visualize predictions
    visualize_price_predictions(
        current_year=datetime.now().year,
        predictions=predictions,
        title="House Price Predictions Over Time"
    )

# %%
if __name__ == "__main__":
    # Train the model
    print("Starting training...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    
    print("\nTraining model...")
    trained_model, _ = train(model, train_loader, optimizer, loss_fn)
    
    print("\nTesting model...")
    test_metrics = test(trained_model, test_loader, loss_fn)
    
    # Save model and artifacts together
    print("\nSaving model and artifacts...")
    model_save = {
        'model_state_dict': trained_model.state_dict(),
        'artifacts': feature_info,
        'test_metrics': test_metrics
    }
    with open('house_price_model.pkl', 'wb') as f:
        pickle.dump(model_save, f)
    print("\nModel and artifacts saved to 'house_price_model.pkl'")
    
    # Test the saved model with house #14
    print("\nTesting saved model functionality...")
    loaded_model = test_saved_model(11)

    # Make predictions for multiple years
    print("\nMaking predictions for multiple years...")
    test_predictions(loaded_model)