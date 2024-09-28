import uuid
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, collection_name):
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def populate_vectors(self, user_input, response):
        combined_text = f"User: {user_input} Assistant: {response}"
        embeddings = self.embedding_model.encode(combined_text).tolist()
        self.collection.add(
        embeddings=[embeddings],
        documents=[combined_text],
        ids=[str(uuid.uuid4())]  
    )

    def search_context(self, query, n_results=1):
        query_embeddings = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=query_embeddings, n_results=n_results)
        if not results['documents']:
            return []

        filtered_results = []
        for document_list in results['documents']:
            for document in document_list:
                filtered_results.append({"document": document})
        return filtered_results