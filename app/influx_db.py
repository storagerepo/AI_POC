# influx_db.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from influxdb_client import InfluxDBClient, Point, WritePrecision
import os
from datetime import datetime

# InfluxDB configuration
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "lvSU-8WG2oV51x0dOQ86oIo8lmWReNSoTlic1yrJAgkBj9wQKUTt6zh_n207DDoEyDb07q_03TjcblQmv7ScNg=="
INFLUXDB_ORG = "Benhive"
INFLUXDB_BUCKET = "initial"
from pydantic import BaseModel

# Create InfluxDB client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG, timeout=60)  # Increase timeout to 60 seconds
router = APIRouter()
class ViewedProperty(BaseModel):
    user_id:int
    property_id:int
    location:str
    property_type:str

def log_user_login(username: str):
    """Log user login details to InfluxDB."""
    try:
        print(username,"fsfs")
        write_api = client.write_api()
        point = (
            Point("user_logins")
            .tag("username", username)
            .field("login_time",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        print(f"Logged login for user: {username}")
    except Exception as e:
        print(f"Failed to log login for user {username}: {e}")

@router.post("/viewed_properties")
def viewedProperties(details:ViewedProperty):
    try:
        write_api=client.write_api()
        point=  point = (
            Point("viewed_properties")
            .tag("user_id", details.user_id)
            .tag("property_id",details.property_id)
            .tag("location",details.location)
             .tag("property_type",details.property_type)
            .field("visited_time",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        write_api.write(bucket=INFLUXDB_BUCKET, record=point)
        return JSONResponse({"status":True,"message":"Added Successfully"})
    except Exception as e:
         return JSONResponse(f"Failed to entry the viewed Property Details: {e}")
    
@router.get("/getViewedPropertiesByUserId")
def get_viewed_properties_by_user_id(user_id: int):
    """Retrieve viewed properties from InfluxDB for a specific user."""
    try:
        query_api = client.query_api()
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
          |> range(start: -1d)  // Fetch data from the last day; adjust as needed
          |> filter(fn: (r) => r._measurement == "viewed_properties" and r.user_id == "{user_id}")
          |> yield(name: "results")
        '''
        results = query_api.query(org=INFLUXDB_ORG, query=query)

        # Extract results into a list
        viewed_properties = []
        for table in results:
            for record in table.records:
                print(record)
                viewed_properties.append({
                    "user_id": record.values.get("user_id"),
                    "property_id": record.values.get("property_id"),
                    "location": record.values.get("location"),
                    "property_type": record.values.get("property_type"),
                     "visited_time": record.get_time().strftime("%Y-%m-%d %H:%M:%S"),    
                                  
                         })

        return JSONResponse({"status": True, "data": viewed_properties})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve viewed properties: {str(e)}")
class PropertyFilter(BaseModel):
    filter_type: str  # Specify 'location' or 'property_type'
    # filter_value: str  # Value to filter by (location or property type)
    time_range: str = "-24h"  # Time range for the query (default is the last 24 hours)

@router.post("/most_viewed_properties")
def get_most_viewed_properties(filter: PropertyFilter):
    try:
        print('pr',filter)
        # Prepare the Flux query
        query = f'''
        from(bucket: "{INFLUXDB_BUCKET}")
            |> range(start: {filter.time_range})
            |> filter(fn: (r) => r._measurement == "viewed_properties")
            |> group(columns: ["{filter.filter_type}"])
            |> count()
            |> yield(name: "count")
        '''

        # Execute the query
        results = client.query_api().query(query)

        # Prepare the response data
        viewed_properties = []
        for table in results:
            for record in table.records:
                viewed_properties.append({
                    filter.filter_type: record.values.get(filter.filter_type),  # Use get_tag for tag values
                    "view_count": record.get_value()  # Get the count from the record
                })

        return JSONResponse(content={"status": True, "data": viewed_properties})

    except Exception as e:
        return JSONResponse(content={"status": False, "detail": str(e)})