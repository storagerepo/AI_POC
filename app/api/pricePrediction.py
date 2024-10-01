from fastapi import APIRouter
import numpy as np
from pydantic import BaseModel
import joblib
import pandas as pd


model = joblib.load("ml/Price1.pkl")

class InputModel(BaseModel):
    bedrooms: int
    sqft: float
    location: str
    bathrooms: int
    halls: int = None
    balconies: int = None
    parking_spaces: int = None
    age_of_property: int = None
    furnishing: str = None
    facing_direction: str = None
    floor_number: int = None
    total_floors: int = None
    has_lift: bool = None
    property_type: str = None

class PredictionResponse(BaseModel):
    predictedPrice: float

router = APIRouter()

@router.post("/pricePredict", response_model=PredictionResponse)
async def make_prediction(input_data: InputModel):
    mandatory_features = ['bedrooms', 'sqft', 'location', 'bathrooms']
    optional_features = ['halls', 'balconies', 'parking_spaces', 'age_of_property',
                         'furnishing', 'facing_direction', 'floor_number', 'total_floors',
                         'has_lift', 'property_type']
    input_data_dict = input_data.dict() 
    input_complete = {feature: input_data_dict.get(feature, np.nan) for feature in mandatory_features + optional_features}
    input_df = pd.DataFrame([input_complete])
    predicted_price = model.predict(input_df)[0]  
    return {"predictedPrice": float(predicted_price)}  
    