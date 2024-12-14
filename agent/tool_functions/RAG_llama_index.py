import os
from llama_index.core import (VectorStoreIndex,SimpleDirectoryReader,StorageContext,load_index_from_storage,Document )
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser

class RAGSystem:
    def __init__(self, data_path="assets", persist_dir="./storage", chunk_size=700,chunk_overlap=50):
        self.data_path = data_path
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size  # Max size of each chunk in characters
        self.chunk_overlap = chunk_overlap
        self.index = None
        self._initialize_system()

    def query(self, question):
        """
        Queries the index and retrieves relevant documents (chunks).
        """
        if not self.index:
            print("Index not initialized. Ensure the database is prepared.")
            return None

        query_engine = self.index.as_query_engine()
        response = query_engine.query(question)
        return response.response if response.response else "No relevant documents found."

    def _initialize_system(self):
        """
        Loads documents from the specified directory and creates or loads an index.
        """
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        Settings.llm = None

        if not os.path.exists(self.persist_dir):
            # If index does not exist, load the documents, chunk them and create the index
            documents = SimpleDirectoryReader(self.data_path).load_data()
            chunked_documents = self._chunk_documents(documents)
            self._create_index(chunked_documents)
            # Persist the index to storage
            self.index.storage_context.persist(persist_dir=self.persist_dir)
        else:
            # If the index exists, load it from the persisted storage
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("Loaded index from existing storage.")

    def _chunk_documents(self, documents):
        """
        Split documents into chunks of text to enable semantic searching over smaller units.
        """
        text_splitter = SimpleNodeParser.from_defaults(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)  # overlap of 50 characters
        chunked_documents = []

        for document in documents:
            # Split each document into chunks
            chunks = text_splitter.split_text(document.text)
            for idx, chunk in enumerate(chunks):
                # Wrap each chunk into a Document object with a unique ID
                chunked_documents.append(Document(text=chunk, doc_id=f"{document.doc_id}_chunk_{idx}"))

        print(f"Created {len(chunked_documents)} chunks from {len(documents)} documents.")
        return chunked_documents

    def _create_index(self, documents):
        """
        Creates a vector-based index from the chunked documents.
        """
        if documents:
            self.index = VectorStoreIndex.from_documents(documents)
            print(f"Indexed {len(documents)} chunks into the vector store.")
        else:
            print("No documents to index.")

# Example Usage
if __name__ == "__main__":
    rag_system = RAGSystem()

    # Query the system
    question = "How can we buy a home ?"
    response = rag_system.query(question)
    print("Response:", response)
