import click
from pathlib import Path
from src.document_processing.processor import DocumentProcessor
from src.core.vectorstore import VectorStore
from src.core.llm import init_llm

class RAGSystem:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = init_llm()

    def process_and_index_document(self, file_path: Path):
        """Process a document and index it in the vector store."""
        # Process the document
        chunks = self.document_processor.process_document(file_path)
        
        # Create vectors and store them
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        self.vector_store.create_vector_store(texts, metadatas)
        self.vector_store.save_vector_store()
        return len(chunks)

    def query_document(self, query: str, k: int = 4):
        """Query the document using similarity search."""
        return self.vector_store.similarity_search(query, k=k)

@click.group()
def cli():
    """RAG CLI System for document querying."""
    pass

@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
def index(file_path):
    """Index a document for searching."""
    rag = RAGSystem()
    file_path = Path(file_path)
    num_chunks = rag.process_and_index_document(file_path)
    click.echo(f"Successfully processed and indexed {num_chunks} chunks from {file_path}")

@cli.command()
@click.argument('query')
@click.option('--k', default=4, help='Number of results to return')
def search(query, k):
    """Search the indexed documents."""
    rag = RAGSystem()
    results = rag.query_document(query, k=k)
    
    click.echo("\nSearch Results:")
    for i, result in enumerate(results, 1):
        click.echo(f"\n--- Result {i} (Score: {result['score']:.4f}) ---")
        click.echo(f"Source: {result['metadata']['source']}")
        click.echo(f"Content: {result['content'][:200]}...")

if __name__ == '__main__':
    cli()