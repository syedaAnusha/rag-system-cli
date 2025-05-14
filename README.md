# RAG CLI System

A Retrieval-Augmented Generation (RAG) Command Line Interface system built using LangChain and Google's Generative AI (Gemini).

## Project Overview

This CLI system implements a RAG pipeline that enhances Large Language Model (LLM) responses by retrieving relevant context from a document collection. It uses Google's Gemini model for both text generation and embeddings.

## Features

- Document processing and ingestion pipeline
- Vector storage for efficient similarity search
- RAG-enhanced responses using Google's Gemini model
- Command-line interface for easy interaction
- Modular and extensible architecture

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

1. Place your documents in the `data/raw` directory
2. Run the CLI:
   ```bash
   python main.py [command] [options]
   ```

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
