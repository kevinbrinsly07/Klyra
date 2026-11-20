# Klyra AI

Klyra AI is a full-stack retrieval-augmented generation (RAG) knowledge assistant. It lets users upload PDF documents, ask context-aware questions, and continue conversations with source references.

## Features

- PDF ingestion with text extraction, chunking, embeddings, and duplicate detection
- Hybrid retrieval using semantic search and BM25 keyword search
- Reciprocal Rank Fusion (RRF) for combining search results
- Context-aware follow-up questions using conversation history
- Source-aware answers generated through the configured OpenAI model
- Conversation history, selection, and persistence
- Document listing, upload, refresh, and deletion workflows
- FastAPI validation and structured error responses
- React loading, success, and error states

## Tech Stack

### Backend

- Python
- FastAPI and Uvicorn
- OpenAI API
- Sentence Transformers
- ChromaDB
- BM25 (`rank-bm25`)
- SQLite-based conversation persistence
- PyPDF for PDF text extraction

### Frontend

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS
- Bun

## Architecture

```text
Next.js frontend
        |
        v
FastAPI REST API
        |
        +--> Conversation service ----> SQLite
        |
        +--> RAG service
                |
                +--> PDF extraction and chunking
                +--> Embeddings ------------> ChromaDB
                +--> BM25 keyword search
                +--> RRF hybrid ranking
                +--> OpenAI response generation
```

## Requirements

- Python 3.10+
- Bun
- An OpenAI API key

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Klyra
```

### 2. Configure the backend

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_actual_api_key
OPENAI_MODEL=gpt-5.5
```

The application fails fast if `OPENAI_API_KEY` is missing.

### 3. Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install frontend dependencies

```bash
cd frontend
bun install
cd ..
```

## Running the Application

Start the backend from the repository root:

```bash
uvicorn app.main:app --reload --port 8000
```

Start the frontend in a second terminal:

```bash
cd frontend
bun dev
```

Open [http://localhost:3000](http://localhost:3000).

The frontend uses `http://localhost:8000` by default. To use another backend URL, create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Check API health and configured model |
| `POST` | `/api/v1/chat` | Ask a question and optionally continue a conversation |
| `GET` | `/api/v1/conversations` | List saved conversations |
| `GET` | `/api/v1/conversations/{conversation_id}/messages` | Get conversation messages |
| `PATCH` | `/api/v1/conversations/{conversation_id}` | Rename a conversation |
| `DELETE` | `/api/v1/conversations/{conversation_id}` | Delete a conversation |
| `POST` | `/api/v1/documents/upload` | Upload and index a PDF |
| `GET` | `/api/v1/documents/` | List indexed documents |
| `DELETE` | `/api/v1/documents/{document_id}` | Delete a document and its indexed chunks |

### Chat example

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Summarize the uploaded documents.",
    "top_k": 3
  }'
```

### Document upload example

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@sample_knowledge.pdf"
```

Only PDF files up to 10 MB are accepted.

## Testing

Run the backend test suite:

```bash
pytest
```

Run frontend linting:

```bash
cd frontend
bun run lint
```

Run a frontend production build:

```bash
cd frontend
bun run build
```

## Project Structure

```text
app/
  core/           Application configuration
  routers/        FastAPI route handlers
  schemas/        Request and response models
  services/       RAG, search, documents, conversations, and LLM logic
  vectorstore/    ChromaDB integration
frontend/
  app/            Next.js application and UI
evaluation/       Retrieval and answer-quality evaluation scripts
data/             Runtime databases, vector data, and uploaded files
test_*.py         Backend unit and integration tests
```

## Retrieval Details

The search pipeline combines embedding-based semantic retrieval with BM25 keyword retrieval. Results are merged with Reciprocal Rank Fusion, then filtered and passed through context processing before answer generation. The current RAG retrieval threshold is configured in `RAGService`, with `top_k` bounded between 1 and 10.

## Notes

- Runtime data is stored under `data/` and should not be committed.
- Uploaded documents are processed temporarily and removed after indexing.
- The backend allows local frontend origins on ports 3000.
- API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) while the backend is running.
