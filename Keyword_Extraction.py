import requests
import numpy as np
import json

def extract_entities_using_llm(user_input):
    model = "mistralai/Mistral-7B-Instruct-v0.3"
    prompt = f"""Extract relevant information from the following real estate query:

Query: "{user_input}"

Please extract:
1. Property location (city or region)
2. Maximum price (in numbers only; do not include words like 'billions' or 'millions')
3. Minimum price (in numbers only; do not include words like 'billions' or 'millions')
4. Type of property (e.g., 'house', 'apartment', 'villa', etc.)
5. Neighborhood (e.g., 'school', 'park', 'theater', 'museum', etc.)
6. Amenities (e.g., 'gym', 'swimming pool', 'garden', etc.)

Return the response in the following format:
{{
    "Location": "<Location>",
    "Max Price": "<MaxPrice>",
    "Min Price": "<MinPrice>",
    "Property Type": "<Type>",
    "Neighborhood": "<Neighborhood>",
    "Amenities": "<Amenities>"
}}"""

    try:
        response = requests.post('https://api.together.xyz/v1/chat/completions', json={
            "model": model,
            "max_tokens": 500,
            "temperature": 0.5,
            "top_p": 0.9,
            "messages": [{"content": prompt, "role": "user"}]
        }, headers={"Authorization": "Bearer dabd7ce421ffdc821514b05906cc4294ef10a171c79eb6e6e03293426ec60d09"})  # Replace with your API key
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return None

# Example of user input
user_input = "Find me properties for a single family in Miami from 2 billion to 4 billion near stadium and vivas school with garden, garage in it."
extracted_info = extract_entities_using_llm(user_input)

# Post-process to handle unspecified fields
if extracted_info:
    try:
        extracted_info = json.loads(extracted_info)
        extracted_info = {
            "Location": extracted_info.get("Location", np.nan),
            "Max Price": float(extracted_info.get("Max Price", np.nan)) if extracted_info.get("Max Price") else np.nan,
            "Min Price": float(extracted_info.get("Min Price", np.nan)) if extracted_info.get("Min Price") else np.nan,
            "Property Type": extracted_info.get("Property Type", np.nan),
            "Neighborhood": extracted_info.get("Neighborhood", np.nan),
            "Amenities": extracted_info.get("Amenities", np.nan),
        }

    except json.JSONDecodeError as e:
        print(f"Error parsing extracted info: {e}")
        extracted_info = {
            "Location": np.nan,
            "Max Price": np.nan,
            "Min Price": np.nan,
            "Property Type": np.nan,
            "Neighborhood": np.nan,
            "Amenities": np.nan
        }
    except Exception as e:
        print(f"Unexpected error: {e}")
        extracted_info = {
            "Location": np.nan,
            "Max Price": np.nan,
            "Min Price": np.nan,
            "Property Type": np.nan,
            "Neighborhood": np.nan,
            "Amenities": np.nan
        }

print(extracted_info)
