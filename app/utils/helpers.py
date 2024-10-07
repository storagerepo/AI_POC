from together import Together
from app.config import TG_API_KEY


together_client = Together(api_key=TG_API_KEY)


def get_embedding(text: str):
    response = together_client.embeddings.create(
        input=text,
        model="togethercomputer/m2-bert-80M-8k-retrieval"
    )
    return response.data[0].embedding