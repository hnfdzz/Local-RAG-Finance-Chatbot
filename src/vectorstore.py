import os
from typing import List
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import config

class VectorStoreManager:
    """Modul pengelolaan Vector Database ChromaDB lokal."""
    
    def __init__(self, persist_directory: str = config.VECTOR_DB_DIR, model_name: str = config.EMBEDDING_MODEL):
        self.persist_directory = persist_directory
        print(f"[INFO] Inisialisasi HuggingFace Embedding Model: {model_name}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vector_store = None

    def initialize_db(self) -> Chroma:
        """Membuka atau menginisialisasi direktori ChromaDB."""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        return self.vector_store

    def add_documents(self, documents: List[Document]) -> Chroma:
        """Menambahkan chunks dokumen baru ke dalam database vektor."""
        if not self.vector_store:
            self.initialize_db()
            
        self.vector_store.add_documents(documents)
        print(f"[INFO] Berhasil menyimpan {len(documents)} vektor ke ChromaDB ({self.persist_directory}).")
        return self.vector_store

    def get_retriever(self, search_type: str = "similarity", k: int = 4):
        """Mengembalikan objek retriever untuk pencarian konteks terdekat."""
        if not self.vector_store:
            self.initialize_db()
        return self.vector_store.as_retriever(search_type=search_type, search_kwargs={"k": k})

    def reset_database(self) -> None:
        """Mereset atau menghapus seluruh isi database vektor lokal."""
        if self.vector_store:
            self.vector_store.delete_collection()
            print("[INFO] Vector database berhasil dikosongkan.")