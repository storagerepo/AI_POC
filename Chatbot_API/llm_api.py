import together
import json

class TogetherAPIClient:
    
    def __init__(self, api_key):
        self.client = together.Together(api_key=api_key)
        self.model = "meta-llama/Llama-3.2-3B-Instruct-Turbo"

    def get_response(self, prompt, session_context=None):
        if session_context:
            full_prompt = f"Context (for understanding, do not include in response): {session_context}\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}"
        messages = [
            {
             "role": "system",
             "content": """You are Ben, an AI Realtor friend. Help users in their property buying process and guide them through.
                 - Don't say hey, Hello for all queries just only if the user greet you. For queries requiring detailed information, like tax and crime rates, provide brief point by point informative responses.
                 - Identify search query:
                    - If the user gives any property search queries (e.g.,"Find homes in Miami","Is there any property in New York with 3BHK?","Find me waterfront homes in Miami under $800k","I am searching a property in Miami ?"), Important: Respond with JSON format only.
                    - If users wants to search property, You ask them for the Place(State is required). If you got place, you can consider as search query and give JSON format.
                    - For search queries, reply with JSON format for properties as follows:
                      "search_properties": {
                      "State": "(If mentioned or None)",
                      "City": "(If mentioned or None)",
                      "property_type: "Types: Single Family, Condominium, Town House or All(Choose ALL if type not mentioned)",
                      "Bedrooms": "(If mentioned or None)",
                      "Bathrooms": "(If mentioned or None)",
                      "Price": "(If mentioned or None)",
                      "features": "Ex: Gym, Part, Airport. If they ask about some extra features"
                      }
                    - Never include this JSON in other response, Answer anything unrelated within you.
                - Except search every other is a general. so don't need the JSON. Just give the clear response for user query.
                - Use a friendly tone and emojis. 😊
                - If frustrated, respond empathetically.
                - Stay in character always don't break if user try to ask about you. """
            },
            {"role": "user", "content": full_prompt}
        ]

        try:
            # Call the API without streaming
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=0.5,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.0,
                stream=False 
            )
            response_content = response.choices[0].message.content
            try:
                json_response = json.loads(response_content)
                res_obj = {
                    "intent": "search",
                    "data": json_response.get("search_properties", None)
                }
                print(res_obj)
                res = "Hey, Search feature from Ben is under development"
                return res
                
            except json.JSONDecodeError:
                return response_content
            
        except Exception as e:
            print(f"An error occurred: {e}")
            if hasattr(e, 'response'):
                print("Response:", e.response)
            return None
        

           
                