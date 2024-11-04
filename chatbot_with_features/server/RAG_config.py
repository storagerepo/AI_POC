from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class RAGSystem:
    def __init__(self, data_path="./assets", chunk_size=1000, chunk_overlap=150):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = None
        self.db = None
        self._prepare_database()

    def query(self, question):
        num_results=3
        search_docs = self.db.similarity_search(question)
        if search_docs:
            return ' '.join(doc.page_content for doc in search_docs[:num_results])
        else:
            print("No relevant documents found.")
            return None

    def _prepare_database(self):
        documents = self._load_documents()
        chunks = self._split_text(documents)
        self._initialize_embeddings()
        self.db = FAISS.from_documents(chunks, self.embeddings)
        print(f"Indexed {len(chunks)} chunks into FAISS database.")

    def _load_documents(self):
        loader = DirectoryLoader(self.data_path, glob="*.txt")
        data = loader.load()
        print(f"Loaded {len(data)} documents from directory '{self.data_path}'.")
        return data

    def _split_text(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def _initialize_embeddings(self):
        model_path = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {'device': 'cpu'}  
        encode_kwargs = {'normalize_embeddings': False} 

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"Initialized embeddings using model '{model_path}'.")


if __name__ == "__main__":
    rag_system = RAGSystem()  
    question = "How Ben works"
    context = rag_system.query(question)
    if context:
        print("Retrieved context for LLM:")
        for i, paragraph in enumerate(context):
            print(f"\nParagraph {i + 1}:\n{paragraph}")


