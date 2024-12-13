import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
time_series_df = pd.read_csv("t1.csv")
property_df = pd.read_csv("p1.csv")

# Preprocess datasets
time_series_df['User ID'] -= time_series_df['User ID'].min()
time_series_df['Property ID'] -= time_series_df['Property ID'].min()
property_df['Property ID'] -= property_df['Property ID'].min()

# Map interaction types to numeric values
interaction_map = {"view": 1, "like": 2, "bid": 3, "searched for": 1, "shared": 1, "favourites": 2}
time_series_df["Interaction Value"] = time_series_df["Interaction Type"].map(interaction_map)
time_series_df = time_series_df[["User ID", "Property ID", "Interaction Value", "Timestamp"]]

# Normalize interaction values
time_series_df["Interaction Value"] = (
    time_series_df["Interaction Value"] - time_series_df["Interaction Value"].min()
) / (time_series_df["Interaction Value"].max() - time_series_df["Interaction Value"].min())

# Add temporal features
time_series_df['Timestamp'] = pd.to_datetime(time_series_df['Timestamp'])
time_series_df['Month'] = time_series_df['Timestamp'].dt.month

# Dataset dimensions
num_users = time_series_df['User ID'].nunique()
num_items = property_df['Property ID'].nunique()

# Split data
train_df, test_df = train_test_split(time_series_df, test_size=0.2, random_state=42)

# Dataset class
class PropertyDataset(Dataset):
    def __init__(self, df):
        self.user_tensor = torch.tensor(df["User ID"].values, dtype=torch.long, device=device)
        self.item_tensor = torch.tensor(df["Property ID"].values, dtype=torch.long, device=device)
        self.target_tensor = torch.tensor(df["Interaction Value"].values, dtype=torch.float32, device=device)

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return len(self.user_tensor)

# DataLoaders
batch_size = 512
train_dataset = PropertyDataset(train_df)
test_dataset = PropertyDataset(test_df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model class
class HybridRecommender(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, hidden_size):
        super(HybridRecommender, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.fc1 = nn.Linear(embedding_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        dot_product = (user_emb * item_emb).sum(dim=1, keepdim=True)
        x = torch.cat([user_emb, item_emb], dim=1)
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.fc2(x)
        return self.sigmoid(x + dot_product)

# Initialize model
embedding_size = min(32, num_users // 10, num_items // 10)
hidden_size = 64
model = HybridRecommender(num_users, num_items, embedding_size, hidden_size).to(device)

# Training setup
criterion = nn.BCEWithLogitsLoss()  # Use binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# Training loop
epochs = 100
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for user, item, target in train_loader:
        optimizer.zero_grad()
        predictions = model(user, item).squeeze()
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    test_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for user, item, target in test_loader:
            predictions = model(user, item).squeeze()
            loss = criterion(predictions, target)
            test_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    # Learning rate scheduler
    scheduler.step()

    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    ndcg = ndcg_score([all_targets], [all_preds])

    print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}, NDCG = {ndcg:.4f}")

# Save model
torch.save(model.state_dict(), "hybrid_recommender.pth")
print("Model saved as 'hybrid_recommender.pth'.")

# Load model
loaded_model = HybridRecommender(num_users, num_items, embedding_size, hidden_size).to(device)
loaded_model.load_state_dict(torch.load("hybrid_recommender.pth"))
loaded_model.eval()
print("Model loaded successfully.")

# Recommendation function with diversity
def recommend_properties(user_id, top_k=5):
    user_tensor = torch.tensor([user_id] * num_items, dtype=torch.long, device=device)
    item_tensor = torch.arange(num_items, dtype=torch.long, device=device)
    with torch.no_grad():
        predictions = loaded_model(user_tensor, item_tensor).squeeze()
    top_indices = torch.topk(predictions, top_k).indices
    recommended_items = top_indices.cpu().numpy()
    recommended_properties = property_df[property_df["Property ID"].isin(recommended_items)]
    return recommended_properties

# Example recommendation
user_id = 99
recommendations = recommend_properties(user_id, top_k=10)
print(f"Top Recommendations for User {user_id}:")
print(recommendations)
