import requests
import os
from dotenv import load_dotenv
load_dotenv()

def get_nearby_information(latitude, longitude, place_types):
    """
    Fetches the top 5 rated places for each specified type near the given coordinates,
    dynamically adjusting the search radius until results are found.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        place_types (list[str]): List of place types (e.g., ['school', 'restaurant']).

    Returns:
        dict: A dictionary with place types as keys and a list of top 5 places as values.
    """
    try:
        API_KEY = os.getenv("GOOGLE_API_KEY")
        initial_radius = 3000
        max_radius = 10000
        increment = 2000
        results = {}

        for place_type in place_types:
            radius = initial_radius
            places = []

            while radius <= max_radius and not places:
                #Get nearby places
                places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius={radius}&type={place_type}&key={API_KEY}"
                places_response = requests.get(places_url)
                places_response.raise_for_status()
                places_data = places_response.json()

                # Extract place IDs, names, and locations
                for place in places_data.get("results", []):
                    place_id = place["place_id"]
                    name = place["name"]
                    location = place["geometry"]["location"]
                    places.append({"place_id": place_id, "name": name, "location": location})

                if not places:
                    print(f"No places of type '{place_type}' found within {radius} meters. Expanding search radius...")
                    radius += increment

            #Get details for each place and add distance
            for place in places:
                place_id = place["place_id"]
                details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,rating,formatted_address,formatted_phone_number,website,geometry&key={API_KEY}"
                details_response = requests.get(details_url)
                details_response.raise_for_status()
                details_data = details_response.json()

                result = details_data.get("result", {})
                place["rating"] = result.get("rating", 0)  # Default to 0 if no rating
                place["address"] = result.get("formatted_address", "N/A")
                place["phone"] = result.get("formatted_phone_number", "N/A")
                place["website"] = result.get("website", "N/A")

                # Add distance using Distance Matrix API
                destination = f'{place["location"]["lat"]},{place["location"]["lng"]}'
                distance_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?origins={latitude},{longitude}&destinations={destination}&key={API_KEY}"
                distance_response = requests.get(distance_url)
                distance_response.raise_for_status()
                distance_data = distance_response.json()

                distance_element = distance_data["rows"][0]["elements"][0]
                place["distance"] = distance_element.get("distance", {}).get("text", "N/A")

            #Sort by rating and limit to top 5 places for this place type
            top_places = sorted(places, key=lambda x: x["rating"], reverse=True)[:5]
            results[place_type] = top_places

            #Filter out 'place_id', 'location', 'address'
            filtered_results = {
                place_type: [
                    {key: value for key, value in place.items() if key not in ('place_id', 'location', 'address')}
                    for place in places
                ]
                for place_type, places in results.items()
            }

        return filtered_results


    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return {}


def get_top_places_by_city_or_state(city_or_state, place_types):
    """
    Fetches the top 5 rated places for each specified type in a given city or state.

    Args:
        city_or_state (str): The city or state name for the search.
        place_types (list[str]): List of place types (e.g., ["school", "restaurant"]).

    Returns:
        dict: A dictionary with place types as keys and a list of top 5 places as values.
    """
    try:
        API_KEY = os.getenv("GOOGLE_API_KEY")
        if not API_KEY:
            raise ValueError("Google API key not found. Set it in your environment variables.")

        results = {}
        for place_type in place_types:
            query = f"{place_type} in {city_or_state}"
            url = f"https://maps.googleapis.com/maps/api/place/textsearch/json?query={query}&key={API_KEY}"
            
            # Fetch initial list of places
            response = requests.get(url)
            response.raise_for_status()
            places_data = response.json()
            
            places = []
            for place in places_data.get("results", []):
                place_id = place["place_id"]
                details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields=name,rating,formatted_address,formatted_phone_number,website&key={API_KEY}"
                details_response = requests.get(details_url)
                details_response.raise_for_status()
                details = details_response.json().get("result", {})

                places.append({
                    "name": details.get("name", "N/A"),
                    "rating": details.get("rating", 0),  # Default to 0 if no rating
                    "address": details.get("formatted_address", "N/A"),
                    "phone": details.get("formatted_phone_number", "N/A"),
                    "website": details.get("website", "N/A"),
                })

            # Sort by rating and limit to top 5
            top_places = sorted(places, key=lambda x: x["rating"], reverse=True)[:5]
            results[place_type] = top_places

        return results

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return {}
    except ValueError as ve:
        print(f"Error: {ve}")
        return {}


# latitude = 41.165238  # Example coordinates
# longitude = -104.759763
# # place_type = "school"
# # place_type = "restaurant"
# # place_type = "university"
# place_type = []"gym"


# top_places = fetch_top_places(latitude, longitude, place_type)
# for place in top_places:
#     print(f"Name: {place['name']}")
#     print(f"Rating: {place['rating']}")
#     print(f"Address: {place['address']}")
#     print(f"Phone: {place['phone']}")
#     print(f"Website: {place['website']}")
#     print(f"Distance: {place['distance']}")
#     print("-" * 50)

if __name__ == "__main__":
    city_or_state = "San Francisco, California"
    place_types = ["airport","school"]

    top_places = get_top_places_by_city_or_state(city_or_state, place_types)
    for place_type, places in top_places.items():
        print(f"--- {place_type.capitalize()} ---")
        for place in places:
            print(f"Name: {place['name']}")
            print(f"Rating: {place['rating']}")
            print(f"Address: {place['address']}")
            print(f"Phone: {place['phone']}")
            print(f"Website: {place['website']}")
            print("-" * 50)