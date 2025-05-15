# RAG CLI System

A Retrieval-Augmented Generation (RAG) Command Line Interface system built using LangChain and Google's Generative AI (Gemini).

## Project Overview

This CLI system implements a RAG pipeline that enhances Large Language Model (LLM) responses by retrieving relevant context from a document collection. It uses Google's Gemini model for both text generation and embeddings.

## Features

- 📚 Document Processing
  - PDF document support with efficient chunking
  - RecursiveCharacterTextSplitter with configurable parameters
  - Metadata preservation (page numbers, chunk numbers)
- 🔍 Advanced Search & Retrieval
  - FAISS vector store for efficient similarity search
  - Google's Gemini model for embeddings
  - Configurable context window (k parameter)
  - Multiple retrieval techniques:
    - Similarity Search: Basic retrieval with stuff chain
    - MMR (Maximum Marginal Relevance): Diverse results with contextual compression
    - Multiple-Query: Comprehensive results by generating related questions
  - 2D UMAP Visualization of semantic space:
    - Shows entire document corpus
    - Highlights retrieved documents (d1, d2, ...)
    - Marks original query (Q) and expanded queries (q1, q2, ...)
- 💡 Question Answering
  - Multiple chain types optimized for different scenarios:
    - ConversationalRetrievalChain with stuff chain for chat sessions
    - RetrievalQA with refine chain for one-off MMR queries
    - Multiple-query expansion for comprehensive answers
  - Contextual answer generation
  - Source attribution with page and chunk references
  - Rich formatted output with code highlighting

## Tech Stack

- **LLM**: Google Gemini (gemini-2.0-flash)
- **Embeddings**: Google Generative AI Embeddings (models/embedding-001)
- **Framework**: LangChain
- **Vector Store**: FAISS
- **Document Processing**: Support for PDF, DOCX, and other text formats

## Project Structure

```
rag-system-cli/
├── src/
│   ├── config/            # Configuration settings
│   ├── core/             # Core RAG components
│   │   ├── llm.py        # LLM configuration
│   │   ├── embeddings.py # Embedding model setup
│   │   └── vectorstore.py# Vector store operations
│   ├── document_processing/ # Document handling
│   └── utils/            # Utility functions
├── data/
│   ├── processed/        # Processed document storage
│   └── raw/             # Raw document storage
├── main.py              # Entry point
└── requirements.txt     # Project dependencies
```

## Setup and Installation

1. Clone the repository
2. Ensure Python 3.8+ is installed
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Index a Document

Index a PDF document to make it searchable:

```bash
python main.py index "documents/your-document.pdf"
```

### 2. Search and Ask Questions

Query the indexed document with questions:

```powershell
python main.py search "your question here"
```

Options:

- `--k`: Number of chunks to use for context (default: 4)

  ```powershell
  python main.py search "what is closure?" --k 5
  ```

- `--search-type`: Retrieval technique to use (default: similarity)

  - `similarity`: Basic similarity search with stuff chain
  - `mmr`: MMR with contextual compression for diverse results
  - `multiple-query`: Comprehensive search using query expansion ```powershell

  # Using MMR with contextual compression

  python main.py search "what is useState hook?" --search-type mmr --k 5

  # Using multiple-query expansion (generates UMAP visualization)

  python main.py search "what is useState hook?" --search-type multiple-query --k 6 --num-queries 3

  ```

  ```

The multiple-query expansion mode generates a UMAP visualization that shows:

- All document chunks in the corpus as blue dots
- Retrieved relevant documents as larger light blue dots (labeled d1, d2, etc.)
- Original query as a red star (labeled Q)
- Expanded queries as orange stars (labeled q1, q2, etc.)

This visualization helps in understanding:

- How the retrieved documents relate to the queries in semantic space
- The distribution of document chunks in the corpus
- The relationship between original and expanded queries

### 3. Interactive Chat Mode

Start an interactive chat session with conversation memory:

```powershell
python main.py chat
```

Options:

- `--k`: Number of chunks to use for context (default: 4)
- `--search-type`: Retrieval technique (default: similarity)
  - `similarity`: Basic search with conversation memory
  - `mmr`: Diverse results while maintaining chat context
  - `multiple-query`: Comprehensive answers with query expansion
- `--num-queries`: Number of expanded queries (for multiple-query mode)
- Chat mode always maintains conversation history

```powershell
# Basic chat with similarity search
python main.py chat --k 4 --search-type similarity

# Chat with MMR for diverse responses
python main.py chat --k 6 --search-type mmr

# Chat with query expansion
python main.py chat --k 6 --search-type multiple-query --num-queries 3
```

The search results include:

- Your question
- A comprehensive answer from the LLM
- Source document information
- Page and chunk references for transparency
- Code snippets with syntax highlighting (when present)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[MIT License](LICENSE)

## Contact

For any questions or feedback, please open an issue in the repository.
