from sentence_transformers import SentenceTransformer
import numpy as np

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    return embedding_model.encode(text).astype(np.float32)

