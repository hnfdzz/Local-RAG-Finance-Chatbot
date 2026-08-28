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
