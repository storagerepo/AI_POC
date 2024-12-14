from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import Any, List, Dict, Optional
from models import Property  # Import your Property model
import json
from db import get_db,SessionLocal
from sqlalchemy import cast
from sqlalchemy.dialects.postgresql import JSONB


def search_properties(property_filter: Dict[str, Optional[Any]] ,db:Optional[Session]) -> Dict[str, List[Dict]]:
        """
        Searches for properties based on the given filter criteria.
        
        Args:
            property_filter (dict): Dictionary containing filter criteria.
            
        Returns:
            dict: A dictionary containing a message and a list of matching properties.
        """
        # Start the query
        # db: Session = get_db()
        query = db.query(Property)

        # Initialize a list to store active filters
        filters = []

        # Dynamically add filters based on the keys provided in the property_filter dictionary
        if 'state' in property_filter:
            filters.append(Property.state == property_filter['state'])
        if 'state_id' in property_filter:
            filters.append(Property.state_id == property_filter['state_id'])
        if 'city' in property_filter:
            filters.append(Property.city == property_filter['city'])
        if 'property_type' in property_filter:
            filters.append(Property.property_type == property_filter['property_type'])
        if 'bedrooms' in property_filter:
            filters.append(Property.bedrooms == property_filter['bedrooms'])
        if 'bathrooms' in property_filter:
            filters.append(Property.bathrooms == property_filter['bathrooms'])
        if 'price_min' in property_filter:
            filters.append(Property.price >= property_filter['price_min'])
        if 'price_max' in property_filter:
            filters.append(Property.price <= property_filter['price_max'])
        
        # Uncomment and implement this feature if necessary
        if 'features' in property_filter:
            filters.append(Property.features.cast(JSONB).op('@>')(json.dumps(property_filter['features'])))
            
        if 'nearby' in property_filter:
            filters.append(Property.nearby.cast(JSONB).op('@>')(json.dumps(property_filter['nearby'])))

        # Apply the filters to the query if any exist
        if filters:
            query = query.filter(and_(*filters))

        # Execute the query and fetch results
        properties = query.all()

        # Serialize results into a list of dictionaries
        serialized_properties = [_to_dict(property) for property in properties]
        

        # Return the list of matching properties with a response message
        if serialized_properties:
            return {"message": "success", "data": serialized_properties}
        else:
            return {"message": "No properties available for the specified criteria", "data": []}


def _to_dict(obj):
        """Convert a SQLAlchemy object to a dictionary."""
        if obj is None:
            return None
        return {column.name: getattr(obj, column.name) for column in obj.__table__.columns}


# Example Usage
# if __name__ == "__main__":


#     # Example filter criteria
#     filter_criteria = {
#         "state_id": 'CA',
#         # "city": "",
#         #"property_type": "Apartment",
#         #"bedrooms": 2,
#         #"bathrooms": 2,
#         #"price_min": 200000,
#         #"price_max": 400000,
#         "features": ["balcony", "pool"],
#         "nearby": ["park",'shopping_mall','school']
#     }

#     # Perform the search
#     with SessionLocal() as db:
#         result = search_properties(property_filter=filter_criteria,db=db)

#     # Output the result
#     print(result)

