import click
from pathlib import Path
from src.document_processing.processor import DocumentProcessor
from src.core.vectorstore import VectorStore
from src.core.llm import init_llm
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
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
        self._current_search_type = None
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
    def setup_qa_chain(self, k: int = 4, search_type: str = "similarity", is_chat: bool = False):
        """Initialize the QA chain with the vector store retriever.
        
        Args:
            k (int): Number of documents to retrieve
            search_type (str): Type of search to use. Either "similarity" or "mmr"
            is_chat (bool): Whether this is a chat session (True) or one-off query (False)
        """
        if not self.vector_store.vector_store:
            self.vector_store.load_vector_store()
        
        if not self.vector_store.vector_store:
            raise ValueError("No vector store available. Please index documents first.")

        # For chat mode or similarity search, use basic similarity retriever
        if is_chat or search_type == "similarity":
            base_retriever = self.vector_store.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        else:  # For one-off queries with MMR
            base_retriever = self.vector_store.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 2,  # Fetch more documents for MMR to choose from
                    "lambda_mult": 0.5  # Control diversity
                }
            )
        
        # Add contextual compression only for one-off MMR queries
        if not is_chat and search_type == "mmr":
            # Create a compressor that uses LLM to extract relevant information
            compressor = LLMChainExtractor.from_llm(self.llm)
            
            # Wrap the base retriever with contextual compression
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever
            )
            
            # For one-off MMR queries, use RetrievalQA with refine chain type
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate
              # Create the initial prompt for the first document
            question_prompt = PromptTemplate(
                template="""
You are a senior React developer helping junior developers understand advanced concepts from the book "Learning React Modern Patterns" by Eve Porcello & Alex Banks.

Use the following context to answer the question. Please:
- Respond in the tone of a React developer
- Use bullet points for clarity
- Keep explanations concise but insightful
- Cite the page number and source if available
- If you don't know the answer, say so instead of making it up

Context: {context}
Question: {question}

Answer: Let's explain this step by step:""",
                input_variables=["context", "question"]
            )

            # Create the refine prompt for subsequent documents
            refine_prompt = PromptTemplate(
                template="""
You are a senior React developer helping junior developers understand advanced concepts.

Here's the original question: {question}

We have already provided the following explanation:
{existing_answer}

Now we have found some additional context: {context}

Please refine the explanation by:
1. Adding any new important information
2. Correcting any inaccuracies
3. Maintaining the bullet-point format
4. Keeping the React developer tone
5. Citing page numbers when available

Updated answer:""",
                input_variables=["question", "existing_answer", "context"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="refine",
                retriever=retriever,
                chain_type_kwargs={
                    "question_prompt": question_prompt,
                    "refine_prompt": refine_prompt,
                    "document_variable_name": "context"
                },
                return_source_documents=True
            )
        else:
            retriever = base_retriever
            # For similarity search, use ConversationalRetrievalChain with stuff chain type
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                memory=self.memory,
                return_source_documents=True,
                verbose=False
            )
        return self.qa_chain          
    def query_document(self, query: str, k: int = 4, search_type: str = "similarity", is_chat: bool = False):
        """Query the document using QA chain for contextual answers.
        
        Args:
            query (str): The question to ask
            k (int): Number of documents to retrieve
            search_type (str): Type of search to use. Either "similarity" or "mmr"
            is_chat (bool): Whether this is a chat session
        """
        # Setup QA chain with the specified k value and search type
        if not self.qa_chain or self._current_search_type != search_type:
            self.setup_qa_chain(k=k, search_type=search_type, is_chat=is_chat)
            self._current_search_type = search_type
        
        # Different chains expect different input keys and return different output keys
        if not is_chat and search_type == "mmr":
            # RetrievalQA chain expects "query" and returns "result"
            response = self.qa_chain.invoke({"query": query})
            answer = response["result"]
        else:
            # ConversationalRetrievalChain expects "question" and returns "answer"
            response = self.qa_chain.invoke({"question": query})
            answer = response["answer"]
            
        return answer, response["source_documents"]

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
@click.option('--search-type', type=click.Choice(['similarity', 'mmr']), default='similarity',
              help='Search technique to use. "similarity" for basic similarity search, "mmr" for MMR with contextual compression')
def search(query, k, search_type):
    """Search and answer questions using the indexed documents."""
    rag = RAGSystem()
    answer, source_docs = rag.query_document(query, k=k, search_type=search_type, is_chat=False)
    
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
        
        # Print references with source name only once
        console.print("\n[cyan bold]References from [/cyan bold]" + f"[cyan]📚 {source_name}:[/cyan]")
        for i, doc in enumerate(source_docs, 1):
            page = doc.metadata.get('page', 'Unknown')
            chunk = doc.metadata.get('chunk', 'Unknown')
            console.print(f"[cyan]{i}.[/cyan] Page {page}, Chunk {chunk}")
    
    console.print("\n" + "="*80)

@cli.command()
@click.option('--k', default=4, help='Number of document chunks to use for generating the answer')
@click.option('--search-type', type=click.Choice(['similarity', 'mmr']), default='similarity',
              help='Search technique to use. "similarity" for basic similarity search, "mmr" for MMR with contextual compression')
def chat(k, search_type):
    """Start an interactive chat session with memory of conversation history."""
    rag = RAGSystem()
    console.print("\n[green]Starting chat session. Type 'exit' to end the conversation.[/green]")
    console.print("[green]The system will remember your conversation history.[/green]")
    console.print(f"[green]Using {search_type} search with {k} documents.[/green]\n")
    
    while True:
        # Get user input
        query = input(click.style("You: ", fg='green', bold=True))
        
        # Check for exit command
        if query.lower() in ['exit', 'quit']:
            console.print("\n[yellow]Ending chat session...[/yellow]")
            break
          # Get response
        answer, source_docs = rag.query_document(query, k=k, search_type='similarity', is_chat=True)
        
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
            
            # Print references with source name only once
            console.print("\n[cyan bold]References from [/cyan bold]" + f"[cyan]📚 {source_name}:[/cyan]")
            for i, doc in enumerate(source_docs, 1):
                page = doc.metadata.get('page', 'Unknown')
                chunk = doc.metadata.get('chunk', 'Unknown')
                console.print(f"[cyan]{i}.[/cyan] Page {page}, Chunk {chunk}")
        
        console.print("\n" + "-"*80 + "\n")

if __name__ == '__main__':
    cli()