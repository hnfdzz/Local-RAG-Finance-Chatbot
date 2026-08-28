# Local RAG Finance Intelligence

A 100% offline, privacy first financial document analysis chatbot powered by Retrieval Augmented Generation (RAG), Ollama (Phi 3 Mini), ChromaDB, and Streamlit.

This application enables users to upload financial PDF statements, extract structured financial insights, view interactive analytics, and check financial health completely locally without sending data to cloud APIs.

## Key Features

* Privacy First Architecture: 100% offline execution ensuring confidential financial data never leaves your local machine.
* Interactive RAG Chatbot: Query financial reports naturally using Ollama (phi3:mini) and LangChain.
* Smart Financial Dashboard: Interactive Plotly charts summarizing income, expenses, and cashflow distribution.
* Financial Health Check and Alerts: Automated warnings for overspending and critical allocation ratios.
* Local Session Management: Save and reload conversation history locally in structured JSON files.
* Source Citation: Verifiable response citations referencing exact document sections.

## Project Structure

```text
Local_RAG_Finance_Chatbot/
  data/
    sample_documents/     (Directory for sample PDF financial statements)
  sessions/               (Local chat history JSON backups)
  src/
    __init__.py           (Package initializer)
    loader.py             (PDF text extraction and document chunking)
    rag_chain.py          (RAG pipeline setup and LLM prompt orchestration)
    utils.py              (Utility functions and citation parsing)
    vectorstore.py        (Vector store manager)
  vector_db/              (ChromaDB persistent store)
  .env.example            (Environment variable template)
  .gitignore              (Git exclusion rules)
  app.py                  (Streamlit main application entrypoint)
  config.py               (Central system configuration file)
  README.md               (Project documentation)
  requirements.txt        (Python project dependencies)

Tech Stack
Frontend: Streamlit

LLM and Embeddings: Ollama (phi3:mini), HuggingFace Sentence Transformers

Vector Database: ChromaDB

Orchestration: LangChain

Visualization: Plotly, Pandas

Document Processing: PyPDF, pdfplumber

Getting Started
1. Prerequisites
Python 3.10+

Ollama installed and running locally.

2. Installation
Clone the repository and set up a virtual environment:

Bash
git clone [https://github.com/hnfdzz/Local-RAG-Finance-Chatbot.git](https://github.com/hnfdzz/Local-RAG-Finance-Chatbot.git)
cd Local-RAG-Finance-Chatbot

python -m venv venv
# On Windows:
venv\Scripts\activate

pip install -r requirements.txt
3. Pull Required LLM Model
Ensure Ollama is running, then pull the lightweight Phi 3 model:

Bash
ollama pull phi3:mini
4. Run the Application
Start the Streamlit web dashboard:

Bash
streamlit run app.py

**Langkah Membuat Commit Bertahap di GitHub Desktop**

1. **Commit 1: Update Dokumentasi README**
   * Simpan perubahan berkas `README.md` di VS Code.
   * Buka GitHub Desktop. Pada kolom Summary di bagian kiri bawah, isi:
     `docs: add professional README with features, architecture, and setup guide`
   * Klik **Commit to main**, lalu klik **Push origin**.

2. **Commit 2: Rapikan Komentar Kode**
   * Buka berkas `config.py` di VS Code, tambahkan baris komentar di bagian atas berkas:
     `# Configuration settings for Local RAG Finance Intelligence`
   * Simpan berkas.
   * Di GitHub Desktop, isi kolom Summary:
     `refactor: add inline code comments and configuration annotations`
   * Klik **Commit to main**, lalu klik **Push origin**.

3. **Commit 3: Perbarui Templat Environment**
   * Buka berkas `.env.example` di VS Code, pastikan isinya rapi:
     ```env
     OLLAMA_MODEL=phi3:mini
     EMBEDDING_MODEL=all_MiniLM_L6_v2
     VECTOR_DB_DIR=vector_db
     ```
   * Simpan berkas.
   * Di GitHub Desktop, isi kolom Summary:
     `chore: update environment configurations and example template`
   * Klik **Commit to main**, lalu klik **Push origin**.

**Pengaturan Akhir di Website GitHub**

Buka halaman repositori di browser, lalu lakukan pembaruan berikut:

1. Klik tombol **Settings** (ikon roda gigi di sebelah kanan judul *About* pada bilah samping kanan).
2. Isikan bagian **Description**:
   `100% Offline Local RAG Financial Chatbot with Interactive Analytics and Health Check using Ollama and Streamlit.`
3. Tambahkan topik pada kolom **Topics**:
   `rag`, `langchain`, `ollama`, `financial analysis`, `streamlit`, `chromadb`, `python`.
4. Klik **Save changes**.
