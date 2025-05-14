import click
from pathlib import Path
from src.document_processing.processor import DocumentProcessor
from src.core.vectorstore import VectorStore
from src.core.llm import init_llm
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from rich.console import Console
from rich.panel import Panel
import re

console = Console()

class RAGSystem:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = init_llm()
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )

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

    def setup_qa_chain(self, k: int = 4):
        """Initialize the Conversational QA chain with the vector store retriever."""
        if not self.vector_store.vector_store:
            self.vector_store.load_vector_store()
        
        if not self.vector_store.vector_store:
            raise ValueError("No vector store available. Please index documents first.")
        
        retriever = self.vector_store.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=False
        )        
        return self.qa_chain

    def query_document(self, query: str, k: int = 4):
        """Query the document using QA chain for contextual answers."""
        # Setup QA chain with the specified k value
        if not self.qa_chain:
            self.setup_qa_chain(k=k)
            
        response = self.qa_chain.invoke({"question": query})  # Changed query to question
        return response["answer"], response["source_documents"]  # Changed result to answer

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
    console.print(f"\n[green]Successfully processed and indexed {num_chunks} chunks from {file_path}[/green]")

@cli.command()
@click.argument('query')
@click.option('--k', default=4, help='Number of document chunks to use for generating the answer')
def search(query, k):
    """Search and answer questions using the indexed documents."""
    rag = RAGSystem()
    answer, source_docs = rag.query_document(query, k=k)
    
    # Print the query
    console.print("\n" + "="*80)
    console.print("[blue]🔍 Question: [/blue]" + f"[green bold]\"{query}\"[/green bold]")
    console.print("="*80)
    
    # Print the answer in a panel with markdown formatting
    answer_panel = Panel(
        answer,
        title="[yellow bold]Answer[/yellow bold]",
        border_style="yellow",
        padding=(1, 2)
    )
    console.print("\n", answer_panel)
    
    # Print source information
    if source_docs:
        # Get the source document name (should be same for all chunks)
        source = source_docs[0].metadata.get('source', 'Unknown source')
        source_name = Path(source).name
        
        console.print("\n[cyan bold]Source Document:[/cyan bold]")
        console.print(f"[cyan]📚 {source_name}[/cyan]")
        
        console.print("\n[cyan bold]References:[/cyan bold]")
        for i, doc in enumerate(source_docs, 1):
            page = doc.metadata.get('page', 'Unknown')
            chunk = doc.metadata.get('chunk', 'Unknown')
            console.print(f"[cyan]{i}.[/cyan] Page {page}, Chunk {chunk}")
    
    console.print("\n" + "="*80)

@cli.command()
@click.option('--k', default=4, help='Number of document chunks to use for generating the answer')
def chat(k):
    """Start an interactive chat session with memory of conversation history."""
    rag = RAGSystem()
    console.print("\n[green]Starting chat session. Type 'exit' to end the conversation.[/green]")
    console.print("[green]The system will remember your conversation history.[/green]\n")
    
    while True:
        # Get user input
        query = input(click.style("You: ", fg='green', bold=True))
        
        # Check for exit command
        if query.lower() in ['exit', 'quit']:
            console.print("\n[yellow]Ending chat session...[/yellow]")
            break
        
        # Get response
        answer, source_docs = rag.query_document(query, k=k)
        
        # Print the answer
        answer_panel = Panel(
            answer,
            title="[yellow bold]Assistant[/yellow bold]",
            border_style="yellow",
            padding=(1, 2)
        )
        console.print("\n", answer_panel)
        
        # Print source information
        if source_docs:
            source = source_docs[0].metadata.get('source', 'Unknown source')
            source_name = Path(source).name
            
            console.print("\n[cyan bold]Sources Used:[/cyan bold]")
            for i, doc in enumerate(source_docs, 1):
                page = doc.metadata.get('page', 'Unknown')
                chunk = doc.metadata.get('chunk', 'Unknown')
                console.print(f"[cyan]{i}.[/cyan] Page {page}, Chunk {chunk}")
        
        console.print("\n" + "-"*80 + "\n")

if __name__ == '__main__':
    cli()