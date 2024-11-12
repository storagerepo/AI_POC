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
        if user_input:
            if response:
                self.chat_history.append({"User": user_input, "Bot": response})
        
        encoding_text = f"{user_input}"
        embeddings = self.embedding_model.encode(encoding_text).tolist()
        conversation_doc = json.dumps({"User": user_input, "Bot": response})
        self.collection.add(
            embeddings=[embeddings],
            documents=[conversation_doc],  
            ids=[str(uuid.uuid4())]  
        )
        print(self.search_context)

    def search_context(self, query):
        query_embeddings = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=query_embeddings, n_results=3)
        if not results['documents'] or not results['documents'][0]:
            return []

        filtered_results = []
        for document_list in results['documents']:
            for document in document_list:
                if document is not None:  
                    conversation_data = json.loads(document)  
                    filtered_results.append({
                        "User": conversation_data.get('User'),
                        "Bot": conversation_data.get('Bot')
                    })
                else:
                    print("Encountered a None document in results.") 

        #Add previous two conversation to make it 5 Conversation for Context
        return filtered_results
       