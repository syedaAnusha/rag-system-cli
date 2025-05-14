from typing import List, Dict
from pathlib import Path
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

    def process_pdf(self, file_path: Path) -> List[Dict]:
        """Process a PDF file and return chunks with metadata."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                text += page.extract_text()

            chunks = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[{
                    "source": str(file_path),
                    "file_type": "pdf"
                }]
            )

            return [{
                "text": chunk.page_content,
                "metadata": chunk.metadata
            } for chunk in chunks]

    def process_document(self, file_path: Path) -> List[Dict]:
        """Process a document based on its file type."""
        if file_path.suffix.lower() == '.pdf':
            return self.process_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
