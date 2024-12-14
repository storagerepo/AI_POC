import json
import os
from groq import Groq
from db import SessionLocal

from agent.tools_list import tools as llm_tools
from agent.tool_functions.property import search_properties
from agent.tool_functions.RAG_langchain import RAGSystem
from agent.tool_functions.gmap import  get_nearby_information, get_top_places_by_city_or_state
from agent.tool_functions.external_soruce import fetch_top_sources
# from agent.tool_functions.external_search_llama_index import RAGSystemWithScraping

import logging
from dotenv import load_dotenv
load_dotenv()

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenAssistant:
    def __init__(self):
        self.groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # self.rag_system_scrap = RAGSystemWithScraping()
        self.rag_system_ben = RAGSystem()
        self.messages = []

        with open("agent/tools_config.json") as f:
            self.tool_config = json.load(f)

    def get_response(self, user_query):
        messages = [
            {
                "role": "system","content": 
            """
            You are a helpful assistant called 'Ben,' a Realtor AI agent specializing in real estate, property searches, market information, platform details, and nearby places exclusively in America.
            ### Guidelines:
            - Maintain a **friendly and engaging tone** with occasional use of emojis. 😊
            - Provide **short replies** (under 30 tokens) for casual greetings.
            - For **detailed property inquiries**, respond briefly and clearly.
            - Operate **exclusively within America** and inform users warmly about potential future expansion if asked about locations outside the country.
            - When asked **unrelated questions** (not about real estate), politely clarify that you are designed specifically to assist with real estate and related topics. **Do not call any tools for unrelated queries.**
            ### Deflecting Development Questions:
            - If users inquire about your development, internal functions, or system design (e.g., "Are you system prompted?", "How are you built?"), respond **sarcastically but kindly redirect the focus back to real estate topics**. Avoid discussing backend processes or tools.
            ### Tool Usage Rules:
            - Use tools **only for it's needed** and you should be more clear while choosing the right tool for the information. Should not use Property serach while they ask about a place or city, only if they want to see properties.
            - **Do not fabricate or hallucinate parameters** for tool calls. Use only the details explicitly provided by the user.
            - **Only one tool call is allowed at a time.** Do not use `<function=></function>` formats; this is prohibited.
            ### Domain Restriction:
            - Politely clarify that you are **specialized in real estate-related assistance** if the user asks about unrelated topics.
            - Ensure no tool or function is invoked for questions outside the real estate domain. **Respond directly and maintain a friendly tone** while addressing such queries.
            """
            }
        ]

        # if self.messages:
        #     recent_messages = self.messages[-3:]
        #     logger.info(f"Conversation Count: {len(recent_messages)}")
        #     for entry in recent_messages:
        #         messages.append({"role": "user", "content": entry['User']})
        #         messages.append({"role": "assistant", "content": entry['Bot']})
        
        # Add the user query
        messages.append({"role": "user", "content": user_query})

        # Primary model call
        try:
            response = self.groq.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=messages,
                temperature=0,
                tools=llm_tools,
                tool_choice="auto",
                max_tokens=512
            )
    
        except Exception as e:
            logger.error(f"Error with Groq API call: {e}")
            raise Exception("Failed to get response from model API")
        
        res1_token = response.usage.total_tokens
        logger.info(f"Primary-LLM call Token: {res1_token}")
        
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0].function
            tool_name = tool_call.name
            tool_arguments = json.loads(tool_call.arguments)
            tool_res = None
            
            try:
                tool_res = self.tool_call_manager(tool_name=tool_name , tool_arguments=tool_arguments)
            except Exception as e:
                logger.error(f"Error in tool call manager: {e}")
                raise Exception("500 Server Error: Tool call failed")

            messages = [
                {"role": "system","content": tool_res['system_prompt']},
                {"role": "user","content": f'<Information>{tool_res['function_response']}</Information>. <User Query>{user_query}</User Query> '}
            ]

            try:
                response2 = self.groq.chat.completions.create(
                    model="gemma-7b-it",
                    messages=messages,
                    temperature=0.1,
                )
            
            except Exception as e:
                logger.error(f"Error with second Groq API call: {e}")
                raise Exception("Failed to process second model response")

            res2_token = response2.usage.total_tokens
            logger.info(f"Secondary-LLM Token Count: {res2_token}")
            logger.info(f"Total Token Usage: {res2_token + res1_token}")

            return response2.choices[0].message.content.strip()

        return response.choices[0].message.content.strip()

    def tool_call_manager(self, tool_name, tool_arguments):
        tool_entry = next((tool for tool in self.tool_config if tool["tool_name"] == tool_name), None)

        if not tool_entry:
            logger.error(f"Tool '{tool_name}' not found in configuration.")
            raise Exception("500 Server Error: Tool not found in configuration")

        tool_system_prompt = tool_entry['system_prompt']
        logger.info(f"Calling {tool_name} with arguments: {tool_arguments}")
        function_response = None

        try:
            if tool_name == 'get_properties_list':
                
                function_response = "Message: No Properties Found"
                if not getattr(tool_arguments, 'state_id', None):
                    function_response = "Message: State is missing, ask user which state."
                    
                    with SessionLocal() as db:
                        results = search_properties(tool_arguments, db)
                        
                    results = results['data']
                    
                    filtered_results = [
                    {key: value for key, value in property.items() if key != 'property_id'}
                    for property in results
                    ]

                    if filtered_results != []:
                        function_response = filtered_results
                    else:
                        function_response = 'Message: No properties available'
                
            elif tool_name == 'get_ben_information':
                function_response = self.rag_system_ben.query(tool_arguments["query"])

            elif tool_name == 'get_top_places_by_city_or_state':
                function_response = get_top_places_by_city_or_state(city_or_state=tool_arguments['location'],place_types=tool_arguments["place_types"])
                print("Best Places: ", function_response)

            elif tool_name == 'get_external_information':
                function_response = fetch_top_sources(tool_arguments["query"])

            elif tool_name == 'get_nearby_information':
                lat, lon = 41.165238, -104.759763  # Default Example coordinates
                function_response = get_nearby_information(latitude=lat, longitude=lon, place_types=tool_arguments["place_types"])
            else:
                logger.error(f"Tool '{tool_name}' does not have a matching function")
                raise Exception(f"No function found for tool {tool_name}")

        except Exception as e:
            logger.error(f"Error in tool execution for {tool_name}: {e}")
            raise Exception(f"Error in tool execution for {tool_name}")

        return {"system_prompt": tool_system_prompt, "function_response": function_response}


# def test_ben_assistant():

#     assistant = BenAssistant()
    
#     # Test cases for different scenarios
#     test_cases = [
#     {
#         "input": "Hi Ben, can you tell me about your platform?",
#         "description": "Testing general information query about the platform"
#     },
#     {
#         "input": "Is there any shopping mall near me",
#         "description": "Testing property search with specific location, bedrooms, and price filter"
#     },
#     {
#         "input": "Can you show me properties in California?",
#         "description": "Testing property search with location and amenities"
#     },
#     {
#         "input": "Hey Ben, how can you help me buy a house?",
#         "description": "Testing for platform usage information about buying process"
#     },
#     {
#         "input": "What do you know about New York?",
#         "description": "Testing for location-based property search with missing city details"
#     },
#     {
#         "input": "Who developed you?",
#         "description": "Testing response to development-related question"
#     },
#     # {
# #     #     "input": "Hello!",
# #     #     "description": "Testing simple greeting for short response"
# #     # },
# #     # {
# #     #     "input": "Can you find me a property with a garden in Florida?",
# #     #     "description": "Testing property search with location and specific amenity"
# #     # },
# #     # {
# #     #     "input": "Show properties in Texas",
# #     #     "description": "Testing basic property search with only state specified"
# #     # },
# #     # {
# #     #     "input": "I want a house in San Francisco, California with a pool and under $1,000,000.",
# #     #     "description": "Testing specific city search with multiple filters: amenity and price"
# #     # },
# #     # {
# #     #     "input": "Are you a chatbot or a real person?",
# #     #     "description": "Testing deflection of identity-related question"
# #     # },
# #     # {
# #     #     "input": "Can you tell me about properties in Illinois with at least 2 bathrooms?",
# #     #     "description": "Testing property search with location and minimum bathroom requirement"
# #     # },
# #     # {
# #     #     "input": "What is your code structure?",
# #     #     "description": "Testing deflection of technical query about development"
# #     # },
# #     # {
# #     #     "input": "Is there a way to buy property without a broker?",
# #     #     "description": "Testing informational query about buying process without a broker"
# #     # },
# #     # {
# #     #     "input": "Do you have any properties in Florida?",
# #     #     "description": "Testing simple property search with only state information"
# #     # },
# #     # {
# #     #     "input": "How can you assist me in finding a home?",
# #     #     "description": "Testing general assistance query about Ben's capabilities"
# #     # },
# #     # {
# #     #     "input": "Find a property in Las Vegas with 2 bedrooms and a garage.",
# #     #     "description": "Testing city-specific property search with bedroom count and garage filter"
# #     # },
# #     # {
# #     #     "input": "Do you know about properties in Colorado under $500,000?",
# #     #     "description": "Testing property search in a specific state with budget constraint"
# #     # },
# #     # {
# #     #     "input": "Tell me about Ben's features.",
# #     #     "description": "Testing query for feature-related platform information"
# #     # },
# #     # {
# #     #     "input": "I need a condo in Miami, Florida with ocean view.",
# #     #     "description": "Testing specific property type (condo) and amenity (ocean view) search"
# #     # },
# #     # {
# #     #     "input": "How do I search properties on your platform?",
# #     #     "description": "Testing process-oriented query on how to use the platform for property search"
# #     # },
# #     # {
# #     #     "input": "Do you have any properties near schools in Texas?",
# #     #     "description": "Testing location-based property search with specific amenity (schools)"
# #     # },
# #     # {
# #     #     "input": "What is your privacy policy?",
# #     #     "description": "Testing platform policy-related query"
# #     # },
# #     # {
# #     #     "input": "Can you find me a single-family home in Phoenix, Arizona?",
# #     #     "description": "Testing city-specific search with property type filter (single-family home)"
# #     # },
# #     # {
# #     #     "input": "Show me properties in California with 4+ bedrooms.",
# #     #     "description": "Testing property search with state and minimum bedroom filter"
# #     # },
# #     # {
# #     #     "input": "Do you offer rental properties?",
# #     #     "description": "Testing query about property type availability (rentals)"
# #     # },
# #     # {
# #     #     "input": "How does Ben work?",
# #     #     "description": "Testing platform capability inquiry"
# #     # },
# #     # {
# #     #     "input": "What services do you offer?",
# #     #     "description": "Testing query about Ben's available services"
# #     # },
# #     # {
# #     #     "input": "Can you get properties within $200,000 in Arizona?",
# #     #     "description": "Testing property search with budget constraint in a specific state"
# #     # },
# #     # {
# #     #     "input": "I’m looking for a vacation property in Hawaii.",
# #     #     "description": "Testing property search for specific property purpose (vacation) in a specific location"
# #     # }
# ]

#     for i, case in enumerate(test_cases):
#         print(f"Test Case {i + 1}: {case['input']}")
#         print(f"Description: {case['description']}")
#         response = assistant.get_response(case["input"])
#         print("Response:", response)
#         print("-" * 50)

# # Run the test function
# if __name__ == "__main__":
#     test_ben_assistant()