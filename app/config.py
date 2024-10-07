import os
from dotenv import load_dotenv


load_dotenv()


TG_API_KEY = os.getenv("TG_API_KEY")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

# PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
# os.makedirs(UPLOAD_DIR, exist_ok=True)