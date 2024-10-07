from together import Together
from app.config import TG_API_KEY, MODEL_NAME
from typing import List, Dict, Any

client = Together(api_key=TG_API_KEY)



def get_model_response(messages: List[Dict[str, Any]]) -> str:
    try:
        prompt = "You are a helpful AI assistant. Please provide a response based on the following conversation:\n\n"
        for msg in messages:
            prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
        prompt += "Assistant: "
    
        response = client.chat.completions.create(
         model=MODEL_NAME,
         messages=messages,
         max_tokens=256,
         temperature=0.7,
         top_p=0.9,
         top_k=50,
         repetition_penalty=1.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in get_model_response: {e}")
        return "I'm sorry, but I couldn't generate a response at this time."