"""Here are the Tools for agent can use
   - get_properties_list
   - get_ben_information
   - get_top_places_by_city_or_state
   - get_nearby_information
   - get_external_information
"""

tools = [{
    "type": "function",
    "function": {
        "name": "get_properties_list",
        "description": "Fetch the properties based on provided filter options, only if they searching for properties to buy. Search for properties based on the user's request. The state parameter MUST be specified. If the user hasn't mentioned the state, ask them to provide it before proceeding. If the user provides a city name, make sure to fill in the state (e.g., 'Arlington' -> 'TX'). Only call the function after the state parameter is filled. If the state is provided, proceed to call the function, then ask about other filters (e.g., price range, property type).",
        "parameters": {
            "type": "object",
            "properties": {
                "state_id": {
                    "type": "string",
                    "description": "The State code in United States of America",
                    "enum": ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
                             "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
                             "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT",
                             "VA", "WA", "WV", "WI", "WY"]
                },
                "city": {
                    "type": "string",
                    "description": "The city in the state"
                },
                "property_type": {
                    "type": "string",
                    "enum": ["House", "Condo", "Apartment", "Townhouse"],
                    "description": "The Property type"
                },
                "bedrooms": {
                    "type": "integer",
                    "description": "The Count of bedrooms"
                },
                "bathrooms": {
                    "type": "integer",
                    "description": "The Count of bathrooms"
                },
                "price_min": {
                    "type": "integer",
                    "description": "The minimum price in the user's specified range, used for queries with a lower budget limit."
                },
                "price_max": {
                    "type": "integer",
                    "description": "The maximum price in the user's specified range, used for queries with an upper budget limit."
                },
                "features": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of additional features like Gym, Park, etc., or an empty list if none mentioned",
                    "enum": ["garage", "garden", "fireplace", "balcony", "laundry", "gym", "pool", "elevator",
                             "rooftop_deck", "parking", "central_heating", "patio", "hardwood_floors"]
                },
                "near_by": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "enum": ["park", "grocery_store", "school", "shopping_mall", "hospital", "restaurant", "theme_park",
                             "bus_station"],
                    "description": "List of nearby amenities like Shops, Airport, etc., or an empty list if none mentioned"
                }
            },
            "required": ["state_id"]
        }
    }
},
    {
        "type": "function",
        "function": {
            "name": "get_ben_information",
            "description": "Retrieve detailed information from Ben's internal documentation using the RAG system to answer questions about the platform, property buying assistance, policies, and usage guidance. This includes responding to questions about Ben’s purpose, features, buying process, and how the platform benefits users. For example, answer questions like 'How does Ben work?', 'What are the benefits of using this platform?', or 'How do I start buying a property here?'. Use this function when the user wants to know about Ben (the platform), company policies, or the broker commission-free system. If the user asks a general question like 'Tell me about yourself,' transform it into a more specific query such as 'How does Ben work?' or 'What are Ben's features?' Always ensure the query is suitable for retrieving information from the RAG system.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Refine and form a concise query based on the user’s question, suitable for retrieving information from the RAG system. For example, if the user asks 'Tell me about yourself,' transform this into an appropriate query like 'How does Ben work?' or 'What are Ben's features?'"
                    }
                },
                "required": ["query"]
            }
        }},
    {
    "type": "function",
    "function": {
        "name": "get_nearby_information",
        "description": "Find the top five highly-rated nearby places of a specific type chosen by the user. This tool should ONLY be used when the user asks for places near their current location or a specific property (e.g., 'What are good restaurants near me?' or 'Show me schools near this property'). Focus on proximity to the specified location. Results include reviews, ratings, addresses, and distances from the provided location. Do not use this tool for general queries about a city or state without a specific location reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "place_types": {
                    "type": "array",
                    "description": "The type of places the user wants to search for nearby. Select one or more from the supported types, e.g., ['cafe', 'school'], ['gym', 'supermarket'].",
                    "items": {
                        "type": "string",
                        "enum": ["airport", "cafe", "church", "gym", "hospital", "library", "movie_theater",
                                 "night_club", "restaurant", "school", "university", "shopping_mall", "supermarket",
                                 "train_station"]
                        }
                    }
                },
            "required": ["place_types"]
        }
    }
    },
    {
    "type": "function",
    "function": {
        "name": "get_top_places_by_city_or_state",
        "description": "Fetch the top-rated places of specified types for a given city or state in America. This tool should ONLY be called when the user explicitly mentions a city or state name in the United States (e.g., 'San Francisco', 'California'). Results include reviews, ratings, addresses, phone numbers, and websites. Do not use this tool for queries without a specific city or state name or for locations outside the U.S.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The name of the city or state in the United States where the user wants to search for places. Example: 'San Francisco', 'California'."
                },
                "place_types": {
                    "type": "array",
                    "description": "The type of places the user wants to search for in the specified location. Select one or more from the supported types, e.g., ['cafe', 'school'], ['gym', 'supermarket'].",
                    "items": {
                        "type": "string",
                        "enum": ["airport", "cafe", "church", "gym", "hospital", "library", "movie_theater",
                                 "night_club", "restaurant", "school", "university", "shopping_mall", "supermarket",
                                 "train_station"]
                    }
                }
            },
            "required": ["location", "place_types"]
        }
    }
    },
    {
        "type": "function",
        "function": {
            "name": "get_external_information",
            "description": "Fetches up-to-date information from the internet for topics requiring recent insights, such as real estate trends, market updates, or other time-sensitive queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Refine and form a concise query based on the user’s question, suitable for retrieving information. The query string the user wants to search on the internet."
                    }
                },
                "required": ["query"]
            }
        }
    }
]
