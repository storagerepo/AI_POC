import os
from bs4 import BeautifulSoup
import requests
import random
from llama_index.core import (VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader, load_index_from_storage)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser

class RAGSystemWithScraping:
    def __init__(self, data_path="agent/tool_functions/assets", persist_dir="./storage", model_path="multi-qa-MiniLM-L6-cos-v1", chunk_size=700,chunk_overlap=50):
        self.data_path = data_path
        self.model_path = model_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = persist_dir
        self.index = None
        self.embedding_model = None
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edge/91.0.864.59",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
        ]
        self._initialize_system()

    def query(self, question, num_results=3):
        """
        Queries the index and retrieves relevant documents.
        """
        self.scrape_and_prepare_index(question)
        
        if not self.index:
            print("Index not initialized. Ensure the database is prepared.")
            return None

        print(f"Querying the index with question: '{question}'")
        response = self.index.as_query_engine(similarity_top_k=num_results).query(question)
        top_result = response.response if response.response else "No relevant documents found."
        return top_result.strip()

    def scrape_and_prepare_index(self, query):
        """
        Scrape web content based on the query and prepare a new index.
        """
        links, contents = self._get_valid_content(query)
        if not contents:
            print("No valid content found for scraping.")
            return

        # Log scraped links
        print(f"Scraped the following links: {links}")

        # Split content into chunks
        all_chunks = []
        for content in contents:
            chunks = self._split_into_chunks(content)
            print(f"Split content into {len(chunks)} chunks.")
            all_chunks.extend(chunks)

        # Create documents and index
        documents = [Document(text=chunk) for chunk in all_chunks]
        self._create_index(documents)

    def _initialize_system(self):
        """
        Prepares the embeddings and index system for the application.
        """
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.model_path)
        Settings.llm = None
        if not os.path.exists(self.persist_dir):
            print("Index not found. Creating a new index.")
            documents = SimpleDirectoryReader(self.data_path).load_data()
            chunked_documents = self._chunk_documents(documents)
            self._create_index(chunked_documents)
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print("Persisted index to storage.")
        else:
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

    def _get_google_search_links(self, query):
        """
        Fetch Google search result links for a query.
        """
        search_query = '+'.join(query.split())
        url = f"https://www.google.com/search?q={search_query}"
        headers = {"User-Agent": random.choice(self.user_agents)}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            links = []
            unwanted_domains = ["google", "facebook", "instagram", "wikipedia", "statista", 'nar', 'redfin', 'zillow']
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href']
                
                if href.startswith('https://') and not any(domain in href for domain in unwanted_domains):
                    links.append(href)
                    
            print("Fetched search links successfully.")
            return list(set(links))
        
        except requests.exceptions.RequestException as e:
            print(f"Error fetching Google search links: {e}")
            return []

    def _scrape_content(self, url):
        """
        Scrape content from a URL.
        """
        headers = {"User-Agent": random.choice(self.user_agents)}
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if p.get_text(strip=True)]
                headers = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3']) if h.get_text(strip=True)]
                return " ".join(headers + paragraphs)
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
        return None

    def _get_valid_content(self, query):
        """
        Fetch valid content from Google search links.
        """
        links = self._get_google_search_links(query)
        valid_content = []
        valid_links = []
        max_sites = 2

        for link in links:
            if len(valid_content) >= max_sites:
                break  # Stop after 2 valid websites
            
            content = self._scrape_content(link)
            
            if content:
                valid_content.append(content)
                valid_links.append(link)

        return valid_links, valid_content

    def _split_into_chunks(self, text):
        """
        Split text into chunks of a specified size.
        """
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size) if text[i:i + self.chunk_size].strip()]

    def _create_index(self, documents):
        """
        Creates a vector-based index from the loaded documents.
        """
        self.index = VectorStoreIndex.from_documents(documents)
        print(f"Indexed {len(documents)} documents into the vector store.")


# Example Usage
if __name__ == "__main__":
    rag_system = RAGSystemWithScraping()

    # Query the system
    question = "Tell me about Damon Salvatore ?"
    response = rag_system.query(question)
    print("Response: \n", response)