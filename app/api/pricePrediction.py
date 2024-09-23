from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("ml/Price_Prediction_model.pkl")

# Define input model without default values
class InputModel(BaseModel):
    sqft: float
    bathrooms: float
    bedrooms: float
    location: str
    halls: int
    balconies: int
    parking_spaces: int
    age_of_property: int
    furnishing: str
    facing_direction: str
    floor_number: int
    total_floors: int
    has_lift: bool
    property_type: str

# Define prediction response model
class PredictionResponse(BaseModel):
    predictedPrice: float

router = APIRouter()

@router.post("/pricePredict", response_model=PredictionResponse)
async def make_prediction(input_data: InputModel):
    # Convert input data to DataFrame for prediction
    input_df = pd.DataFrame([input_data.dict()])

    # Predict price
    prediction = model.predict(input_df)

    # Return prediction
    return PredictionResponse(predictedPrice=prediction[0])
