import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing 
import pickle

# Import data
houses = pd.read_csv('Property Recommendation/tamil_nadu.csv')

# Select numerical data features
houses_a = houses[['BEDS', 'BATHS', 'SQ_FT', 'FLOORS', 'LAT', 'LONG']]
house_data_c = houses_a.values.tolist()

# Process neighborhood and style as text features
neighbors = houses['NEIGHBORHOOD'].tolist()
styles = houses['STYLE'].tolist()

# One-hot encode
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
        return houses.loc[top10_index_array, :]
    return pd.DataFrame()

print(recommend_top10_2_1_2([10, 1, 55])) 

# Save the model
with open('recommendation_model.pkl', 'wb') as model_file:
    pickle.dump((df_normalized, houses), model_file)