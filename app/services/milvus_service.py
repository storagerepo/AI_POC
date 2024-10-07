from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from app.config import MILVUS_HOST, MILVUS_PORT


class MilvusService:
    def __init__(self):
        self.collection_name = "chat_history"
        self.dimension = 768
        self.connect()
        
    def connect(self):
            try:
               connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
               print("Successfully connected to Milvus")
            except Exception as error:
                  print(f"Failed to connect to Milvus: {error}")
        
    def create_collection(self):
        try:
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="user_message", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="ai_response", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)
                ]
                schema = CollectionSchema(fields, "Chat history for similarity search")
                collection = Collection(self.collection_name, schema)
                
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 1024}
                }
                collection.create_index("embedding", index_params)
                print(f"Collection '{self.collection_name}' created successfully")
            else:
                collection = Collection(self.collection_name)
                print(f"Collection '{self.collection_name}' already exists")
            
            collection.load()
        except Exception as e:
            print(f"Error in create_collection: {e}")
      
        
        
    def insert_chat(self, user_message, ai_response, embedding):
        try:
            collection = Collection(self.collection_name)
            data = [
                [user_message],
                [ai_response],
                [embedding]
            ]
            insert_result = collection.insert(data)
            collection.flush()
            print(f"Inserted data with ID: {insert_result.primary_keys[0]}")
            return insert_result.primary_keys[0]
        except Exception as e:
            print(f"Error in insert_chat: {e}")
            return None
        
    def search_similar_chats(self, query_embedding, limit=5):
        try:
            collection = Collection(self.collection_name)
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["id", "user_message", "ai_response"]
            )
            return [
                {
                    "id": hit.entity.id,
                    "user_message": hit.entity.user_message,
                    "ai_response": hit.entity.ai_response,
                    "distance": hit.distance
                }
                for hit in results[0]
            ]
        except Exception as e:
            print(f"Error in search_similar_chats: {e}")
            return []
        
    def get_chat_by_id(self, chat_id):
        try:
            collection = Collection(self.collection_name)
            result = collection.query(
                expr=f"id == {chat_id}",
                output_fields=["id", "user_message", "ai_response"]
            )
            if result:
                return {
                    "id": result[0]['id'],
                    "user_message": result[0]['user_message'],
                    "ai_response": result[0]['ai_response']
                }
            return None
        except Exception as e:
            print(f"Error in get_chat_by_id: {e}")
            return None
        
    def get_conversation_context(self, query_embedding, limit=3):
        try:
            collection =  Collection(self.collection_name)
            search_params = {
                "metric_type": "L2",
                "params": {
                    "nprobe": 10,
                }
            }
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=[
                    "user_message",
                    "ai_response",
                ]
            )
            context = []
            for hit in results[0]:
              context.append({
                "user_message": hit.get('user_message'),
                "ai_response": hit.get('ai_response'),
            })
            return context
        except Exception as error:
            print(f"Error in get_conversation_context: {error}")
            return []
    
    
milvus_service = MilvusService()