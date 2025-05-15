import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from umap.umap_ import UMAP
from rich.console import Console

console = Console()

def plot_query_document_space(queries, query_embeddings, documents, doc_embeddings, corpus_docs, corpus_embeddings, output_dir: Path):
    """Generate a 2D scatter plot showing embeddings distribution using UMAP.
    
    Args:
        queries (list): List of query strings (original + expanded)
        query_embeddings (np.array): Embeddings of all queries
        documents (list): List of retrieved document chunks
        doc_embeddings (np.array): Embeddings of retrieved documents
        corpus_docs (list): List of all documents in corpus
        corpus_embeddings (np.array): Embeddings of all documents in corpus
        output_dir (Path): Directory to save the plot
    """
    console.print("[blue]Generating UMAP visualization...[/blue]")
    
    # Combine all embeddings for UMAP
    all_embeddings = np.vstack([query_embeddings, doc_embeddings, corpus_embeddings])
      # Configure and run UMAP
    reducer = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1
    )
    points = reducer.fit_transform(all_embeddings)
    
    # Split points back into respective groups
    n_queries = len(queries)
    n_docs = len(documents)
    
    query_points = points[:n_queries]
    doc_points = points[n_queries:n_queries + n_docs]
    corpus_points = points[n_queries + n_docs:]
      # Create the plot
    plt.style.use('classic')  # Use classic style instead of seaborn
    plt.figure(figsize=(12, 12), facecolor='white')
    
    # Set up the plot style
    plt.gca().set_facecolor('white')      # Plot data points first without labels
    # Corpus documents as background
    plt.scatter(corpus_points[:, 0], corpus_points[:, 1], 
               c='green', s=100, alpha=0.9)
    
    # Retrieved documents
    plt.scatter(doc_points[:, 0], doc_points[:, 1],
               c='lightblue', s=100, alpha=0.6, zorder=2)
    
    # Add labels for retrieved documents
    for i, point in enumerate(doc_points):
        plt.annotate(f'd{i+1}', (point[0], point[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12)
      # Original query only
    plt.scatter(query_points[0, 0], query_points[0, 1],
               c='red', marker='*', s=100,
               edgecolor='white', linewidth=1, zorder=4)
    plt.annotate('Q', (query_points[0, 0], query_points[0, 1]),
                xytext=(5, 5), textcoords='offset points',
                color='red', fontsize=12, fontweight='bold')
                  # Create custom legend handles
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D    
    legend_elements = [
        # Corpus dot (using scatter for the legend)
        plt.scatter([], [], c='green', s=100, alpha=0.9, label='corpus'),
        
        # Retrieved doc dot
        plt.scatter([], [], c='lightblue', s=100, alpha=0.6, label='retrieved docs'),
        
        # Original query star
        plt.scatter([], [], c='red', marker='*', s=100, label='Query')
    ]
    
    plt.title('Semantic Space Visualization (UMAP)', fontsize=14, pad=20)
    plt.grid(True, alpha=0.4)
    plt.legend(scatterpoints=1, fontsize=10, framealpha=1)
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f'embeddings_umap_{timestamp}.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close()
    
    console.print(f"[green]Visualization saved to:[/green] {output_path}")
    return output_path
