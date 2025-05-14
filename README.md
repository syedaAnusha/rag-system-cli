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
- 💡 Question Answering
  - LangChain's RetrievalQA chain
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

```bash
python main.py search "your question here"
```

Options:

- `--k`: Number of chunks to use for context (default: 4)
  ```bash
  python main.py search "what is closure?" --k 10
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
