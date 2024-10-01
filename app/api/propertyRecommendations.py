from fastapi import APIRouter, HTTPException
import numpy as np
from pydantic import BaseModel
import pickle
import pandas as pd
import os

# Pydantic input and output models
class InputModel(BaseModel):
    viewedProperties: list[int]

class PropertyModel(BaseModel):
    ID: int
    CITY: str
    ADDRESS: str
    BEDS: int
    BATHS: int
    HALF_BATH: int
    SQ_FT: int
    STYLE: str
    FLOORS: int
    RECENT_PRICE: int
    LAT: float
    LONG: float
    NEIGHBORHOOD: str

class RecommendationResponse(BaseModel):
    recommendProperties: list[PropertyModel]

# Check if the model file exists
if not os.path.isfile('ml/recommendation_model.pkl'):
    raise FileNotFoundError("The model file 'recommendation_model.pkl' is not found.")

# Load the model
with open('ml/recommendation_model.pkl', 'rb') as model_file:
    ls, houses, df_normalized = pickle.load(model_file)

# Function to select normalized data based on house ID
def data_select_nor(id):
    if id in houses['ID'].values:  # Check if ID exists in the dataset
        index_number = list(houses['ID']).index(id)
        house_select = ls[index_number]
        return house_select
    else:
        raise ValueError(f"House ID {id} not found in the dataset.")

# Function to calculate cosine similarity between two vectors
def cosine_similarity(v1, v2):
    dot_x1_x2 = np.dot(v1, v2)
    x1 = np.array(v1)
    x2 = np.array(v2)
    n1_n = np.linalg.norm(x1)
    n2_n = np.linalg.norm(x2)
    return dot_x1_x2 / (n1_n * n2_n) if n1_n != 0 and n2_n != 0 else 0

# Generate a list of similarities between target house and all others
def recommend_list_nor(target, exclude_ids):
    similarities = []
    for i in range(len(ls)):
        if houses['ID'].iloc[i] not in exclude_ids:  # Exclude the input IDs
            item = []
            item.append(i)
            item.append(cosine_similarity(target, ls[i]))
            similarities.append(item)
    return similarities

# Router for FastAPI
router = APIRouter()

@router.post("/recommendation", response_model=RecommendationResponse)
def propertyRecommendationTopTen(input_data: InputModel):
    frames = []
    id_array = input_data.viewedProperties  # Extract the list of viewedProperties
    for id in id_array:
        try:
            target = data_select_nor(id)
            result = recommend_list_nor(target, id_array)  
            result_list = pd.DataFrame.from_records(result)
            result_list = result_list.sort_values(by=[1], ascending=False)
            top10 = result_list.head(10)
            frames.append(top10)
        except ValueError as e:
            print(f"Error: {e}")  # Log the missing ID error
            continue  # Skip the missing ID and continue with other IDs
    
    if frames:
        total = pd.concat(frames)
        total = total.sort_values(by=[1], ascending=False).drop_duplicates(subset=[0])
        res = total.head(10)
        top10_index_array = list(res[0])
        df = houses.loc[top10_index_array, :]

        # Convert the DataFrame to a list of PropertyModel
        property_list = df.to_dict(orient='records')  # Convert to list of dictionaries
        return {"recommendProperties": property_list}

    raise HTTPException(status_code=404, detail="No properties found.")
