import uuid
import chromadb
from sentence_transformers import SentenceTransformer
import json

class VectorStore:
    def __init__(self, collection_name):
        self.embedding_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)

        #Array for Storing conversation, until we use Database
        self.chat_history = []

    def populate_vectors(self, user_input, response):
        #Added chat history to save last conversations
        self.chat_history.append({"User": user_input, "Bot": response})
        
        encoding_text = f"{user_input}"
        embeddings = self.embedding_model.encode(encoding_text).tolist()
        conversation_doc = json.dumps({"User": user_input, "Bot": response})
        self.collection.add(
            embeddings=[embeddings],
            documents=[conversation_doc],  
            ids=[str(uuid.uuid4())]  
        )

    def search_context(self, query):
        query_embeddings = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=query_embeddings, n_results=3)
    
        # Return empty list if no documents are found
        if not results['documents'] or not results['documents'][0]:
            return []

        filtered_results = []
    
        # Flatten and filter results from ChromaDB
        for document_list in results['documents']:
            for document in document_list:
                if document:  # Avoid None documents
                    conversation_data = json.loads(document)
                    filtered_results.append({
                        "User": conversation_data.get('User'),
                        "Bot": conversation_data.get('Bot')
                    })
    
        # Add previous conversations for context
        convo_count = len(self.chat_history)
    
        if convo_count >= 2:
            filtered_results.extend(self.chat_history[-2:])  # Last two conversations
        elif convo_count == 1:
            filtered_results.append(self.chat_history[-1])   # Only the last conversation
    
        return filtered_results
        