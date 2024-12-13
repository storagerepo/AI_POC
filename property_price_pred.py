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
EPOCHS = 500
PATIENCE = 50
LEARNING_RATE = 0.0001  

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

# Find numerical and categorical features
total_features = df.columns
print(f"Total features available in the dataset: {len(total_features)}")

numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
print(f"The numerical features available for prediction: {numeric_features}\nand the total count is: {len(numeric_features)}")

categorical_features = df.select_dtypes(include=['object']).columns
print(f"The categorical features available for prediction: {categorical_features}\nand the total count is: {len(categorical_features)}")

# %%
# Feature selection based on correlation
correlations = df.select_dtypes(include=['int64', 'float64']).corr()['SalePrice']
numerical_features = correlations[
    (correlations > 0.5) &
    (correlations.index != 'SalePrice')
].index.tolist()

categorical_features = df.select_dtypes(include=['object'])\
    .columns[df.select_dtypes(include=['object']).nunique() < 20].tolist()[:5]

print("\nAutomatically selected features:")
print("Top numerical features:", numerical_features)
print("Top categorical features:", categorical_features)

# %%
# Handle missing values
print("\nHandling missing values:")
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())
df[categorical_features] = df[categorical_features].fillna(df[categorical_features].mode().iloc[0])

# Encode categorical features
label_encoders = {}
for c in categorical_features:
    label_encoders[c] = LabelEncoder()
    df[c] = label_encoders[c].fit_transform(df[c])
    print(f"\nEncoding for {c}:")
    for i, category in enumerate(label_encoders[c].classes_):
        print(f"{category} -> {i}")

# %%
# Add time feature
def add_time_feature(df):
    current_year = datetime.now().year
    df = df.copy()
    df['YearsFromNow'] = 0
    return df, current_year

df, current_year = add_time_feature(df)

# %%
# Prepare features and target
X = df[numerical_features + categorical_features + ['YearsFromNow']]
y = df['SalePrice']

# Scale numerical features and target
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Scale the target variable (important for neural networks)
y_scaler = StandardScaler()
y = pd.Series(y)  # Ensure y is a pandas Series
y = y_scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# %%
# Create Dataset class
class HousePriceDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to tensors directly
        if isinstance(X, pd.DataFrame):
            self.X = torch.FloatTensor(X.values).to(device)
        else:
            self.X = torch.FloatTensor(X).to(device)
            
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            self.y = torch.FloatTensor(y.values).reshape(-1, 1).to(device)
        else:
            self.y = torch.FloatTensor(y).reshape(-1, 1).to(device)
    
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
        
        self.model = nn.Sequential(
            # First layer with larger size
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            
            # Second layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            
            # Third layer with residual connection
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.1),
            
            # Fourth layer
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size//2),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(hidden_size//2, 1),
            nn.Softplus()  # Ensure positive outputs
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize model, loss, and optimizer
model = HousePriceModel(input_size=len(X.columns)).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    """
    Predict house price for a future year with enhanced trend stability
    """
    # Create a copy of features to avoid modifying the original
    future_features = features.copy()
    
    # Update the time-based feature
    years_from_now = future_year - current_year
    future_features['YearsFromNow'] = years_from_now
    
    # Convert to tensor and make prediction
    with torch.inference_mode():
        if isinstance(future_features, pd.Series):
            features_tensor = torch.FloatTensor(future_features.values).unsqueeze(0).to(device)
        else:
            features_tensor = torch.FloatTensor(future_features).unsqueeze(0).to(device)
        
        prediction = model(features_tensor)
        price = y_scaler.inverse_transform(prediction.cpu().numpy())[0][0]
    
    # Get base price (current price)
    base_price = y_scaler.inverse_transform([[y_test[0]]])[0][0]
    
    # Apply trend stability with smoothing
    if years_from_now > 0:
        # Calculate minimum and maximum allowable prices
        min_yearly_change = 0.98  # Max 2% decrease per year
        max_yearly_change = 1.12  # Max 12% increase per year
        
        # Calculate compound growth bounds
        min_price = base_price * (min_yearly_change ** years_from_now)
        max_price = base_price * (max_yearly_change ** years_from_now)
        
        # Apply exponential smoothing for more stable predictions
        alpha = 0.7  # Smoothing factor
        smoothed_price = alpha * price + (1 - alpha) * base_price * (1.05 ** years_from_now)
        
        # Ensure price stays within bounds
        price = max(min(smoothed_price, max_price), min_price)
    
    return max(price, 10000)  # Ensure minimum reasonable price

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
    
    return plt.gcf()

# %%
# Train the model
print("Starting training...")
trained_model, final_loss = train(
    model=model,
    train_loader=train_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHS,
    patience=PATIENCE
)
print(f"\nTraining finished with best loss: {final_loss:.4f}")

# %%
# Test the model
print("\nEvaluating model...")
test_results = test(
    model=trained_model,
    test_loader=test_loader,
    loss_fn=loss_fn
)

# %%
# Example prediction for a house
print("\nExample prediction for a house:")
if isinstance(X_test, pd.DataFrame):
    sample_house = X_test.iloc[0].values
else:
    sample_house = X_test[0]

with torch.inference_mode():
    sample_tensor = torch.FloatTensor(sample_house).unsqueeze(0).to(device)
    sample_pred = trained_model(sample_tensor)
    sample_pred = y_scaler.inverse_transform(sample_pred.cpu().numpy())[0][0]
    actual_price = y_scaler.inverse_transform([[y_test[0]]])[0][0]
print(f"Predicted price: ${sample_pred:,.2f}")
print(f"Actual price: ${actual_price:,.2f}")

# %%
# Predict future prices for a sample house
print("\nPredicting house prices for the next 5 years...")
predictions = {}
sample_house = X_test.iloc[0]

for year in range(current_year + 1, current_year + 6):  
    predicted_price = predict_future_price(
        trained_model, sample_house, year, current_year
    )
    predictions[year] = predicted_price
    print(f"Predicted price for {year}: ${predicted_price:,.2f}")

# Visualize predictions
visualize_price_predictions(
    current_year=current_year,
    predictions=predictions,
    title="House Price Prediction Over Next 5 Years"
)

# Save the plot
plt.savefig('price_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization has been saved as 'price_predictions.png'")