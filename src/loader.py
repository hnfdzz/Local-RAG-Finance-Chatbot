import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

class DocumentProcessor:
    """Modul untuk memuat dan memproses dokumen PDF keuangan secara terstruktur."""
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Memuat dokumen PDF menggunakan PDFPlumber untuk fleksibilitas pembacaan tabel."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File dokumen tidak ditemukan pada path: {file_path}")
        
        try:
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            
            # Tambahkan metadata kustom jika diperlukan
            for doc in documents:
                doc.metadata["source_file"] = os.path.basename(file_path)
            
            return documents
        except Exception as e:
            # Fallback ke PyPDFLoader jika PDFPlumber menemui hambatan pembacaan
            print(f"[WARN] PDFPlumber gagal ({e}), mengalihkan ke PyPDFLoader...")
            loader = PyPDFLoader(file_path)
            return loader.load()

    def process_and_split(self, file_path: str) -> List[Document]:
        """Memuat PDF dan memotongnya menjadi chunks yang siap di-embed."""
        raw_documents = self.load_pdf(file_path)
        chunked_documents = self.text_splitter.split_documents(raw_documents)
        print(f"[INFO] Dokumen {os.path.basename(file_path)} berhasil diproses menjadi {len(chunked_documents)} chunks.")
        return chunked_documents