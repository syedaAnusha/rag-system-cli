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

    def query_document(self, query: str, k: int = 3):
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
@click.option('--k', default=3, help='Number of results to return')
def search(query, k):
    """Search the indexed documents."""
    rag = RAGSystem()
    results = rag.query_document(query, k=k)
    
    click.echo("\n" + "="*80)
    click.echo(click.style(f"🔍 Search Results for: ", fg='blue') + 
              click.style(f'"{query}"', fg='green', bold=True))
    click.echo("="*80)
    
    for i, result in enumerate(results, 1):
        # Calculate relevance score percentage (converting distance to similarity)
        relevance = ((1 - result['score']) * 100)
        
        # Format the section header
        click.echo(f"\n{click.style(f'Result {i}', fg='blue', bold=True)}")
        click.echo(f"{click.style('Relevance:', fg='cyan')} {relevance:.1f}%")
        
        # Source information
        source = result['metadata'].get('source', 'Unknown source')
        page = result['metadata'].get('page', 'Unknown page')
        chunk = result['metadata'].get('chunk', i)  # Use i as fallback if chunk number not available
        click.echo(f"{click.style('Source:', fg='cyan')} {source}")
        click.echo(f"{click.style('Page:', fg='cyan')} {page}")
        click.echo(f"{click.style('Chunk:', fg='cyan')} {chunk}")
        
        # Content section
        click.echo(f"\n{click.style('Content:', fg='yellow', bold=True)}")
        
        # Clean and format the content
        content = result['content'].strip()
        
        # Remove multiple newlines and spaces
        content = ' '.join([line.strip() for line in content.split('\n') if line.strip()])
        
        # Format code blocks if they exist (text between backticks or with specific indentation)
        if '```' in content or any(line.startswith('    ') for line in content.split('\n')):
            # Split into text and code blocks
            blocks = content.split('```')
            formatted_blocks = []
            for j, block in enumerate(blocks):
                if j % 2 == 1:  # Code block
                    formatted_blocks.append(click.style("\n📝 Code Example:", fg='green', bold=True))
                    formatted_blocks.append(click.style(block.strip(), fg='white', bg='black'))
                else:  # Text block
                    formatted_blocks.append(block.strip())
            content = '\n'.join(formatted_blocks)
        
        # Add ellipsis if content is too long
        if len(content) > 600:
            # Try to find a good breaking point (end of sentence)
            break_point = content[:600].rfind('.')
            if break_point == -1:
                break_point = 600
            content = content[:break_point + 1] + "..."
        
        # Add indentation for better readability
        content = '\n'.join('    ' + line for line in content.split('\n'))
        click.echo(content + '\n')
        
        click.echo("-"*80)

if __name__ == '__main__':
    cli()