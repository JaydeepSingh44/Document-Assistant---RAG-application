Document Assistant is a Retrieval‑Augmented Generation (RAG) application built with LangChain, Mistral AI, and ChromaDB.
It allows users to upload large documents (like PDFs), split them into chunks, embed them, and query them intelligently using an LLM.

---This project demonstrates how to build a scalable assistant that can summarize, answer questions, and provide insights from long documents---

## Tech Stack
-Python 3.10+

-LangChain (prompt templates, runnables, retrievers)

-Mistral AI (embeddings + LLM)

-ChromaDB (vector store for semantic search)

-dotenv (environment variable management)

-PyPDFLoader (document ingestion)

-RecursiveCharacterTextSplitter (chunking large documents)


## Features
- PDF Loader → Ingests documents like books or reports.
- Text Splitter → Breaks documents into manageable chunks.
- Embeddings + Vector Store → Stores chunks for semantic search.
- Retriever + LLM → Answers queries based on relevant chunks.
- Persistent Storage → Saves embeddings locally in chroma-db
  

Project Structure
RAG/
│── .env                  # API keys (MISTRAL_API_KEY)
│── requirements.txt      # Dependencies
│── create_vector_db.py   # Build vector database from PDF
│── main.py               # Query assistant
│── TestBook.pdf          # Sample document (~200 pages)
│── chroma-db/            # Persistent vector store
anf
