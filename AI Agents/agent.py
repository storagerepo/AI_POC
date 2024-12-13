
import requests
import json
import hashlib
import time
import torch
import torch.nn.functional as F
from tools import *
from langchain_community.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModel, pipeline

# Set the API Key for Groq
os.environ['GROQ_API_KEY'] = "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"
from pinecone import Pinecone, ServerlessSpec
import os

# Initialize Pinecone client
api_key = '153347a9-2400-4c8c-94c4-203db308659e'
pc = Pinecone(api_key=api_key)

# Create or get the 'akashakash' index
index_name_akash = "aaa"
if index_name_akash not in pc.list_indexes().names():
    pc.create_index(
        name=index_name_akash,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

pinecone_index_akash = pc.Index(index_name_akash)


# Initialize sentiment analysis pipeline
# sentiment_analyzer = pipeline("sentiment-analysis")

# Initialize sentence-transformers model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Define the Chatbot class with Pinecone context
class Chatbot:
    def __init__(self, model='llama3-70b-8192', temperature=0.1):
        self.llm = ChatGroq(model_name=model, temperature=temperature)
        self.system_prompt = """
You are BEN, a specialized AI real estate assistant. Focus only on real estate queries, responding politely and concisely, with emojis where appropriate. 
1. **tavily_search**: Use this tool only to get real time information about real estate like tax rate, crime rates, and other relevant information.
  
2. **predict_price_tool**:
   - **When to use**: Trigger this tool only when the user specifically asks for property price predictions. The user must provide details such as location, property size, number of bedrooms, and bathrooms. Dont trigger this tool often.
   - **How to call**: Provide all the relevant inputs given by the user (e.g., size, location, bedrooms, bathrooms) to generate a property price prediction. Ask for missing details if necessary.

3. **Mortgage_Calculator**:
   - **When to use**: Use this tool when the user wants to calculate mortgage details, such as monthly payments. The user must specify the loan amount, interest rate, and duration.
   - **How to call**: Ensure you have all the inputs (loan amount, interest rate, duration). If any detail is missing, ask the user for it. Provide clear instructions for the user to input accurate information.

4. **Nearby_Places**:
   - **When to use**: Use this tool when the user asks to find top-rated places near a specific location. The user should provide latitude, longitude, and a radius (Convert miles to meters).
   - **How to call**: Pass the provided latitude, longitude, and radius to the tool. If the user does not give complete location details, ask for them before proceeding.

5. **fetch_company_info**:
   - **When to use**: Use this tool when the user asks about BenHive, the company behind BEN. The user should provide the company name.
   - **How to call**: Pass the complete input to the tool. If the user does not give complete details, ask for them before proceeding.

Guidelines:
- Answer real estate queries (e.g., property prices, mortgages, schools) clearly and avoid unrelated topics.
- Use external tools when required {self.tools} to retrieve property price predictions, calculate mortgage details, find nearby schools, or locate nearby places. Do not use tools for unrelated queries.
- If user seems purchasing property, Gather necessary details (location, budget, property type) and confirm with 'Here are the properties',
- If uSER ASK ANYTHING ABOUT BEN AND BENHIVE, RESPOND WITH PRIDE. REFER THIS TOOL fetch_company_info FOR MORE DETAILS
- Respond in the user's language; do not reference tool usage or previous conversations.
- FOR SOME CASES THERE MIGHT BE POINT BY POINT LIKE NAME, RATING, ADDRESS, PHONE NUMBER, etc. FOR THIS GIVE THE EXACT LINE BY LINE RESPONSE. DONT MISS THE RATINGS AND OTHER INFORMATIONS TOO. 
- Don't avoid the fetched details from tools. Read user query and give the fetched details where applicable
Avoid:
- Discussing non-real estate topics, irrelevant jokes, or singing.
- Mentioning other companies except Benhive.
- Forcing users into decisions (e.g., asking if they plan to buy a property soon).

For Benhive-specific questions, highlight its achievements and services.
"""


#         self.system_instructions = (
#     f"You are BEN, a specialized AI real estate assistant. Answer real estate-related queries concisely. "
#     f"Answer real estate-related queries concisely and with relevant emojis where appropriate even in other languages too. Don't use ugly emojis eg:🤔. "
#     f"Dont tell any joke which is not related to real estate. Dont sing any songs. You are a real estate assistant. Behave properly, be polite, and divert them back to real estate. "
#     f"Do not push the user with questions like 'Are yo  u looking to buy a property soon?' "
#     f"If the user asks for funny facts or jokes about Benhive, respond with pride about our achievements and interesting perspectives instead, for eg: 'While I don't have specific funny facts about Benhive, I can share an interesting perspective: Benhive's approach to democratizing real estate investment is like giving everyone a chance to own a piece of the real estate pie without having to eat the whole thing! It's a bit like being able to enjoy a slice of cake without worrying about the calories of the entire dessert.' "
#     f"Dont tell anything about other companies like Cognizant, Infosys, etc., except Benhive. "
#     f"If a user is searching properties, gather necessary information (Location, Budget, Bedroom) and briefly confirm with 'Here are some properties for you.' "
#     f"Dont force the user like 'By the way, are you looking to buy a property soon?' "
#     f"If asked about Ben, respond with pride, highlighting specifications, mortgage details, etc. "
#     f"Whatever the input is, respond in the user's language. "
#     f"Dont mention old chat in the response."
# )
        self.tools = [
            predict_price_tool,
            mortgage_calculator_tool,
            # Nearby_Places,
            nearby_places,
            tavily_search,
            fetch_company_info
        ]

    def generate_embeddings(self, text):
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1).squeeze().tolist()

    def run(self, input_text):
        # Retrieve past context from Pinecone
        vector = self.generate_embeddings(input_text)
        past_context = ""
        try:
            query_response = pinecone_index_akash.query(
                vector=vector,  # Convert input to vector as needed
                top_k=5,
                include_values=True,
                include_metadata=True
            )
            for match in query_response['matches']:
                past_context += f"{match.get('metadata', {}).get('text', '')}\n"
        except Exception as e:
            print(f"Error querying Pinecone index: {e}")

        # Get sentiment analysis
        # sentiment = sentiment_analyzer(input_text)[0]['label'].lower()

        # Prepare assistant prompt
        assistant_prompt = ChatPromptTemplate.from_messages(
            [
                # SystemMessage(content=f"{self.system_prompt} {self.system_instructions}"),
                SystemMessage(content=f"{self.system_prompt}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
                HumanMessage(content=f"{past_context} {input_text}")
            ]
        )
        print("past_context:", past_context)
        print("input_text:", input_text)
        # Create agent and executor
        assistant_agent = create_openai_tools_agent(self.llm, self.tools, assistant_prompt)
        agent_executor = AgentExecutor(agent=assistant_agent, tools=self.tools, verbose=True)

        # Execute the agent with user input
        result = agent_executor.invoke({'input': input_text})

        if 'output' in result:
            last_response = result['output'].strip()
            unique_id = hashlib.md5(input_text.encode('utf-8')).hexdigest() + f"-{int(time.time())}"
            combined_entry = f"User: {input_text} | Response: {last_response}"

            # Upsert into Pinecone
            pinecone_index_akash.upsert([{
                "id": unique_id,
                "values": vector,  # Use embeddings for the vector field
                "metadata": {
                    "combined_entry": combined_entry,
                    "user_message": input_text,
                    "bot_response": last_response
                }
            }])
            return last_response
        else:
            print("Error: Result does not contain 'output'. Check response structure.")
            return "Sorry, I couldn't process your request."





# input("predict the price of a house with 2 bedroom and 3 bathroom in new york with 1500 sq feet") 