# AG-Based PDF Question Answering System

This project is a **Retrieval-Augmented Generation (RAG)** application that allows you to upload PDF documents, index them using vector embeddings, and ask natural language questions. The system uses **HuggingFace Sentence Transformers** for embedding and **Google Gemini 1.5 Flash** as the language model. It supports multiple interfaces: CLI, FastAPI (for APIs), and Streamlit (for GUI).

---

## Features

- Upload PDFs and ingest them into Qdrant (vector DB)
- Query document content using LLM
- Interfaces:
  - `app.py` – local script for testing
  - `main.py` / `fast_api_app.py` – REST API using FastAPI
  - `query_gui.py` – GUI app using Streamlit

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ramzanniaz331/rag_article_code
cd rag-pdf-app
```

### Folder structure
```
rag_project/
│
├── app.py
├── main.py
├── query_gui.py
├── requirements.txt
├── .env
│
├── uploads/             # <--- Place your PDFs here
│
├── src/
│   ├── ingestion.py
│   ├── retrieval.py
│   └── utils.py

```

### 2. Create and Activate Virtual Environment

```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
### 3. Create and Activate Virtual Environment
```
pip install -r requirements.txt

```

### 4. Install and Run Qdrant Locally

Install qdrant from 

https://qdrant.tech/documentation/quick-start/


Now: 
API Key & Environment Variables

### 5. Get a Gemini API Key

Visit Google AI Studio

Generate an API key for Gemini 1.5 Flash

### 6. Create a .env File in Project Root

GOOGLE_API_KEY=your_gemini_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=your_qdrant_collection_name_here


### 7. PDF Uploads and Path Configuration

Create a folder in project directory and store your pdfs in that

Important:  Update the pdf folder path in the code files (app.py, query_gui.py, etc.) accordingly:


### 8. How to Run

#### 1. Run app.py (for testing queries)

```
python app.py

```
Runs a predefined query

Indexes your documents and prints the LLM’s response in the terminal

#### 2. Run main.py

```
python main.py

```

Open Swagger docs: http://localhost:8000/docs

Use curl/Postman to:

POST PDFs to /upload_pdfs

POST a query to /query

#### 3. Run query_gui.py (Streamlit GUI)

```
streamlit run query_gui.py

```

Upload PDFs using the sidebar

Ask questions via text input

See LLM responses on-screen

