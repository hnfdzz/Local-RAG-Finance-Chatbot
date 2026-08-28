import os

# Alur Direktori Proyek
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "sample_documents")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

# Model AI Lokal (Teroptimasi untuk CPU & RAM Laptop ASUS)
LLM_MODEL = "phi3:mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Parameter Pemotongan Teks (Chunking)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100