from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from models import Property
from db import get_db
from sqlalchemy import and_
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from typing import List, Dict, Any, Optional

router = APIRouter()

class PropertyCreate(BaseModel):
    state: str
    city: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[int] = 0
    bathrooms: Optional[int] = 0
    description: Optional[str] = ""
    price: Optional[float] = 0.0
    price_range: Optional[str] = ""
    features: Optional[List[str]] = None
    nearby: Optional[List[str]] = None
    year_built: int
    amount: float

class PropertySearchParams(BaseModel):
    state: Optional[str] = None
    city: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    price_min: Optional[float] = None  # Minimum price for range search
    price_max: Optional[float] = None  # Maximum price for range search
    year_built: Optional[int] = None
    features: Optional[List[str]] = None
    nearby: Optional[List[str]] = None


class PropertyRead(BaseModel):
    property_id: int
    state: str
    city: Optional[str] = None
    property_type: Optional[str] = None
    bedrooms: Optional[int] = 0
    bathrooms: Optional[int] = 0
    description: Optional[str] = ""
    price: Optional[float] = 0.0
    price_range: Optional[str] = ""
    features: Optional[List[str]] = None
    nearby: Optional[List[str]] = None
    year_built: int
    amount: float

    class Config:
        orm_mode = True

@router.post("/create_property", response_model=PropertyRead)
def create_property(property: PropertyCreate, db: Session = Depends(get_db)):
    # Create a new Property instance using the SQLAlchemy model
    new_property = Property(
        state=property.state,
        city=property.city,
        property_type=property.property_type,
        bedrooms=property.bedrooms,
        bathrooms=property.bathrooms,
        description=property.description,
        price=property.price,
        price_range=property.price_range,
        features=property.features,
        nearby=property.nearby,
        year_built=property.year_built,
        amount=property.amount
    )
    
    db.add(new_property)
    db.commit()
    db.refresh(new_property)
    
    # Return the created property as a PropertyRead instance
    return new_property

@router.post("/search_properties")
def search_properties(
    params: PropertySearchParams,  # Accepting the request body as PropertySearchParams
    db: Session = Depends(get_db)  # Database session
):
    # Start with a query on the Property model
    query = db.query(Property)

    # Apply filters based on provided search parameters
    filters = []
    if params.state:
        filters.append(Property.state == params.state)
    if params.city:
        filters.append(Property.city == params.city)
    if params.property_type:
        filters.append(Property.property_type == params.property_type)
    if params.bedrooms is not None:
        filters.append(Property.bedrooms == params.bedrooms)
    if params.bathrooms is not None:
        filters.append(Property.bathrooms == params.bathrooms)
    if params.price_min is not None:
        filters.append(Property.price >= params.price_min)
    if params.price_max is not None:
        filters.append(Property.price <= params.price_max)
    if params.year_built is not None:
        filters.append(Property.year_built == params.year_built)
    if params.features:
        filters.append(Property.features.contains(params.features))
    if params.nearby:
        filters.append(Property.nearby.contains(params.nearby))

    # Apply the filters to the query
    if filters:
        query = query.filter(and_(*filters))

    # Execute the query and fetch results
    properties = query.all()

    # Return the list of matching properties
    if properties:
        # Return success message and the list of properties
        return {
            "message": "success",
            "data": properties
        }
    else:
        # Return no properties available message
        return {
            "message": "No properties available",
            "data": []
        }