import requests
import json
import joblib
import pandas as pd
import numpy as np
from langchain_community.tools import tool
from pydantic import BaseModel, Field, ValidationError

# Load the trained model
loaded_model = joblib.load('./pickle/Price1.pkl')

# Define the prediction function
def predict_with_mandatory_values(model, input_data):
    mandatory_features = ['bedrooms', 'sqft', 'location', 'bathrooms']
    optional_features = ['halls', 'balconies', 'parking_spaces', 'age_of_property', 
                         'furnishing', 'facing_direction', 'floor_number', 'total_floors', 
                         'has_lift', 'property_type']
    
    all_features = mandatory_features + optional_features
    
    # Fill missing values with NaN
    input_complete = {feature: input_data.get(feature, np.nan) for feature in all_features}
    input_df = pd.DataFrame([input_complete])  # Convert the input to a DataFrame
    
    # Predict the price using the model
    prediction = model.predict(input_df)
    return prediction[0]  

# Tool to predict the price of a property
@tool("Predict_Price")
def predict_price_tool(input):
    """
    Predict the price of a property based on user-provided details.
    """
    predicted_price = predict_with_mandatory_values(loaded_model, input)
    return predicted_price

# Tool to extract entities related to property search
@tool("extract_entities_using_llm")
def extract_entities_using_llm(input):
    """
    Extract key details about property search, such as location, price, bedrooms, bathrooms, and property type.
    """
    model = "mistralai/Mistral-7B-Instruct-v0.3"
    prompt = f"""
    Analyze the user's query to extract key details about property search. Focus on identifying information like location, maximum price (numerical), number of bedrooms, number of bathrooms, and property type.
    User Input: "{input}"
    {{"Location": "<Extracted location>", "MaximumPrice": "<Extracted price>", 
    "Bedrooms": "<Extracted bedrooms>", "Bathrooms": "<Extracted bathrooms>", 
    "PropertyType": "<Extracted property type>"}}
    """
    
    response = requests.post(
        'https://api.together.xyz/v1/chat/completions',
        json={
            "model": model,
            "max_tokens": 500,
            "temperature": 0.5,
            "top_p": 0.9,
            "messages": [{"content": prompt, "role": "user"}]
        },
        headers={"Authorization": f"Bearer dabd7ce421ffdc821514b05906cc4294ef10a171c79eb6e6e03293426ec60d09"}
    )

    if response.status_code == 200:
        try:
            content = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
            return json.loads(content) if content else None
        except json.JSONDecodeError:
            return None
    return None



# Tool to calculate mortgage payments
@tool("Mortgage_Calculator")
def mortgage_calculator_tool(loan_amount, interest_rate, duration_years):
    """
    This tool calculates monthly mortgage payments based on loan amount, interest rate, and loan duration.
    Example: Mortgage Calculator - Loan: $200,000, Rate: 3.5%, Duration: 30 years
    """
    api_url = f"https://api.api-ninjas.com/v1/mortgagecalculator?loan_amount={loan_amount}&interest_rate={interest_rate}&duration_years={duration_years}"
    try:
        response = requests.get(api_url, headers={'X-Api-Key': 'aulT3mgtmm4C2+4be9KNDw==zH0l84hogTSOwDJp'})
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"API request error: {e}"

# Define input schema
class NearbyPlacesInput(BaseModel):
    latitude: float
    longitude: float
    radius: int
    place_type: str 
# Tool to find nearby places
@tool("Nearby_Places")
def nearby_places(input: NearbyPlacesInput):
    """
    Find top-rated places near a given location within a specified radius.
    """
    # Extract inputs
    latitude = input.latitude
    longitude = input.longitude
    radius = input.radius
    place_type = input.place_type
    api_key = "AIzaSyB8dwiq6XBn1Wa-plh0-yZetlzCQLQm9us"  # Replace with your actual API key

    if not api_key:
        return {"error": "API key is missing. Please configure the API key."}

    # Step 1: Fetch nearby places
    try:
        places_url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
            f"location={latitude},{longitude}&radius={radius}&type={place_type}&key={api_key}"
        )
        places_response = requests.get(places_url)
        places_response.raise_for_status()  # Raise HTTP errors
        places_data = places_response.json()

        if "results" not in places_data or not places_data["results"]:
            return {"error": f"No {place_type}s found near the provided location."}
    except requests.RequestException as e:
        return {"error": f"Failed to fetch places data: {str(e)}"}

    # Step 2: Fetch details for each place
    places = []
    for place in places_data.get("results", []):
        try:
            place_id = place.get("place_id")
            if not place_id:
                continue

            details_url = (
                f"https://maps.googleapis.com/maps/api/place/details/json?"
                f"place_id={place_id}&fields=name,rating,formatted_address,"
                f"formatted_phone_number,website&key={api_key}"
            )
            details_response = requests.get(details_url)
            details_response.raise_for_status()
            details_data = details_response.json()

            result = details_data.get("result", {})
            places.append({
                "name": result.get("name", "N/A"),
                "rating": result.get("rating", 0),
                "address": result.get("formatted_address", "N/A"),
                "phone": result.get("formatted_phone_number", "N/A"),
                "website": result.get("website", "N/A"),
            })
        except requests.RequestException as e:
            # Log and continue with the next place
            print(f"Error fetching details for place ID {place.get('place_id')}: {str(e)}")
            continue

    # Step 3: Sort places by rating (descending) and return the top 5
    return sorted(places, key=lambda x: x["rating"], reverse=True)[:5]


import requests

from langchain_community.tools import tool
from tavily import TavilyClient

from langchain_community.tools import tool
from tavily import TavilyClient

# Tool to perform search using Tavily API
@tool("Tavily_Search")
def tavily_search(query: str) -> str:
    """
    Use this tool only to get real time information about real estate like tax rate, crime rates, and other relevant information.

    """
    try:
        # Initialize the Tavily client
        tavily_client = TavilyClient(api_key="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")

        # Execute the search query
        response = tavily_client.search(query)

        # Validate and process the response
        if 'results' in response and response['results']:
            formatted_results = "\n".join(
                [
                    f"{idx + 1}. {result.get('title', 'No Title')} - {result.get('url', 'No URL')}"
                    for idx, result in enumerate(response['results'])
                ]
            )
            return f"Search Results:\n{formatted_results}"
        else:
            return "No results found for your query. Please try a different search term."

    except Exception as e:
        return f"An error occurred during the search: {str(e)}"
    



import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, pipeline
import os
from pinecone import Pinecone, ServerlessSpec

# Initialize the tokenizer and model for generating embeddings
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# Mean pooling function to generate embeddings
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Function to generate embeddings for a given text
def generate_embeddings(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask']).squeeze().tolist()

@tool("Fetch_Company_Info")
def fetch_company_info(input_data):
    """
    This tool queries Pinecone to fetch Ben and Benhive company information based on the input data.
    """
    past_context = ""
    try:
        # Check if the input_data is a dictionary and contains the expected key
        if isinstance(input_data, dict):
            # Extract 'input_data' or 'company_name' from the dictionary
            input_text = input_data.get('input_data') or input_data.get('company_name')
            if not input_text:
                return "No valid company name or input data provided for querying."

        elif isinstance(input_data, str):
            # If input_data is just a string (i.e., no dictionary), use it directly
            input_text = input_data
        else:
            return "Invalid input format. Expected a string or dictionary with 'input_data' or 'company_name'."

        # Generate embedding from the input text
        input_vector = generate_embeddings(input_text)
        
        # Initialize Pinecone client with your API key
        api_key = '153347a9-2400-4c8c-94c4-203db308659e'  # Replace with your environment variable for API key
        pc = Pinecone(api_key=api_key)

        # Assuming `companypolicy` is your Pinecone index name
        index_name = "companypolicy"  # Replace with your actual index name
        
        # Ensure index exists, if not create it (use this only once or as needed)
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1536,  # Set the dimension size of your embeddings
                metric='euclidean',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
        
        # Query Pinecone
        index = pc.Index(index_name)  # Access the specific index

        # Query Pinecone to get similar entries
        query_response = index.query(
            vector=input_vector,  # Querying with the vector generated from user input
            top_k=1,
            include_values=False,  # We only need metadata
            include_metadata=True
        )
        
        # Check if the query response is structured as expected
        if isinstance(query_response, dict) and 'matches' in query_response:
            # Format the query results
            for match in query_response['matches']:
                metadata = match.get('metadata', {})
                # Fetch the combined_entry if it exists
                data = metadata.get('text', 'No data available')
                # Add each combined entry to the past context
                past_context += f"{data}\n"
        else:
            return f"Unexpected query response format: {query_response}"

    except Exception as e:
        return f"Error querying Pinecone index: {e}"
    
    return past_context






















# # # Tool to find nearby schools
# # @tool("Nearby_Schools")
# # def nearby_schools(input: NearbySchoolsInput):
# #     """
# #     Find top-rated schools near a given location within a specified radius.
# #     """
# #     # Extract inputs
# #     latitude = input.latitude
# #     longitude = input.longitude
# #     radius = input.radius

# #     # API configuration
# #     place_type = "school"
# #     api_key = "AIzaSyB8dwiq6XBn1Wa-plh0-yZetlzCQLQm9us"  # Replace with your actual API key

# #     if not api_key:
# #         return {"error": "API key is missing. Please configure the API key."}

# #     # Step 1: Fetch nearby schools
# #     try:
# #         places_url = (
# #             f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?"
# #             f"location={latitude},{longitude}&radius={radius}&type={place_type}&key={api_key}"
# #         )
# #         places_response = requests.get(places_url)
# #         places_response.raise_for_status()  # Raise HTTP errors
# #         places_data = places_response.json()

# #         if "results" not in places_data or not places_data["results"]:
# #             return {"error": "No schools found near the provided location."}
# #     except requests.RequestException as e:
# #         return {"error": f"Failed to fetch schools data: {str(e)}"}

# #     # Step 2: Fetch details for each school
# #     schools = []
# #     for place in places_data.get("results", []):
# #         try:
# #             place_id = place.get("place_id")
# #             if not place_id:
# #                 continue

# #             details_url = (
# #                 f"https://maps.googleapis.com/maps/api/place/details/json?"
# #                 f"place_id={place_id}&fields=name,rating,formatted_address,"
# #                 f"formatted_phone_number,website&key={api_key}"
# #             )
# #             details_response = requests.get(details_url)
# #             details_response.raise_for_status()
# #             details_data = details_response.json()

# #             result = details_data.get("result", {})
# #             schools.append({
# #                 "name": result.get("name", "N/A"),
# #                 "rating": result.get("rating", 0),
# #                 "address": result.get("formatted_address", "N/A"),
# #                 "phone": result.get("formatted_phone_number", "N/A"),
# #                 "website": result.get("website", "N/A"),
# #             })
# #         except requests.RequestException as e:
# #             # Log and continue with the next place
# #             print(f"Error fetching details for place ID {place.get('place_id')}: {str(e)}")
# #             continue

# #     # Step 3: Sort schools by rating (descending) and return the top 5
# #     return sorted(schools, key=lambda x: x["rating"], reverse=True)[:5]








