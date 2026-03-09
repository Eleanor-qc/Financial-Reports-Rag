# Financial Reports RAG

A Retrieval-Augmented Generation (RAG) system that enables question answering over SEC 10-K financial reports.

This project extracts relevant sections from financial filings, converts them into vector embeddings, indexes them with FAISS, and retrieves the most relevant context to answer user questions using a large language model.

The application is deployed through a Streamlit interface and packaged with Docker for reproducibility.

---

# Architecture

The system follows a standard RAG pipeline:

User Query
↓
Streamlit App (app.py)
↓
RAG Engine (rag_core.py)
↓
Embed Query (Vertex AI Embedding Model)
↓
FAISS Vector Search
↓
Retrieve Relevant Chunks
↓
LLM Generates Answer

---

# Data Processing Pipeline

Before the application can answer questions, the financial reports are processed through several steps:

SEC 10-K PDFs
↓
Chunk Extraction (build_chunks.py)
↓
Text Chunks + Metadata
↓
Embedding Generation (Vertex AI)
↓
Vector Index Construction (build_faiss_index.py)
↓
FAISS Index

---

# Repository Structure

Financial-Reports-Rag/
│
├── app.py                 # Streamlit web interface
├── rag_core.py            # Retrieval + LLM answer generation
│
├── build_chunks.py        # Extract and chunk text from 10-K PDFs
├── build_faiss_index.py   # Generate embeddings and build FAISS index
├── pipeline.py            # End-to-end data processing pipeline
│
├── chunks_text.pkl        # Generated text chunks
├── chunks_metadata.pkl    # Metadata for each chunk
│
├── Alphabet 10-K/         # Alphabet 10-K reports
├── Amazon 10-K/           # Amazon 10-K reports
├── Microsoft 10-K/        # Microsoft 10-K reports
├── Oracle 10-K/           # Oracle 10-K reports
│
├── data/                  # FAISS index and processed artifacts
│
├── Dockerfile             # Docker container configuration
├── requirements.txt       # Python dependencies
---

# Tech Stack

- Python
- FAISS (vector similarity search)
- Google Vertex AI (embeddings + LLM)
- Streamlit (web interface)
- Docker (reproducible environment)

---

# How It Works

1. Financial reports (PDFs) are parsed and split into structured text chunks.
2. Each chunk is converted into an embedding using Vertex AI.
3. The embeddings are indexed using FAISS for efficient similarity search.
4. When a user asks a question:
   - The query is embedded
   - The FAISS index retrieves the most relevant chunks
   - The LLM generates an answer using the retrieved context.

---

# Run the Project

## Build the Docker image

docker build -t financial-rag .

## Run the container

docker run -p 8080:8080 financial-rag

## Open the application

http://localhost:8080

---

# Example Questions

Example queries the system can answer:

- What factors drove AWS revenue growth?
- What risks does Microsoft highlight in its annual report?
- How does Alphabet describe its advertising business?
- What are Oracle's key cloud strategy initiatives?

---

# License

This project is for educational and research purposes.
