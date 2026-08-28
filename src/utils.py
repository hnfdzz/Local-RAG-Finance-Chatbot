import re
from typing import List, Dict, Any

def format_currency_idr(amount: float) -> str:
    """Mengubah format angka murni menjadi representasi Rupiah (Rp)."""
    return f"Rp {amount:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def extract_citations(source_documents: List[Any]) -> List[Dict[str, Any]]:
    """Ekstraksi rujukan sumber halaman PDF untuk kebutuhan verifikasi pengguna."""
    citations = []
    for doc in source_documents:
        metadata = getattr(doc, "metadata", {})
        page = metadata.get("page", 0) + 1  # Konversi index 0-based ke nomor halaman asli
        source = metadata.get("source_file", "Dokumen Keuangan")
        citations.append({
            "source": source,
            "page": page,
            "snippet": doc.page_content[:150] + "..."
        })
    return citations

def sanitize_text(text: str) -> str:
    """Membersihkan karakter berlebih dari ekstraksi PDF."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()