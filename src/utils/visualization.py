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
    
    # Calculate actual distances between query and documents
    from scipy.spatial.distance import cdist
    original_query = query_embeddings[0].reshape(1, -1)  # Just use original query
    distances = cdist(original_query, doc_embeddings, metric='cosine').flatten()
    
    # Sort documents by distance
    sort_indices = np.argsort(distances)
    doc_embeddings = doc_embeddings[sort_indices]
    documents = [documents[i] for i in sort_indices]
    
    # First fit UMAP on corpus embeddings
    reducer = UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine'  # Use cosine similarity as it's what we use for retrieval
    )
    
    # Fit and transform corpus points
    corpus_points = reducer.fit_transform(corpus_embeddings)
    
    # Now transform query and retrieved docs using the same reducer
    # This preserves the relative distances between query and retrieved docs
    query_points = reducer.transform(original_query)
    doc_points = reducer.transform(doc_embeddings)
    
    # Add connections between query and top documents
    lines_x = []
    lines_y = []
    for point in doc_points:
        lines_x.extend([query_points[0, 0], point[0], None])
        lines_y.extend([query_points[0, 1], point[1], None])    # Create the plot
    plt.style.use('classic')  # Use classic style instead of seaborn
    plt.figure(figsize=(12, 12), facecolor='white')
    
    # Set up the plot style
    plt.gca().set_facecolor('white')
    
    # Plot connecting lines first (dashed lines)
    plt.plot(lines_x, lines_y, 'k--', alpha=0.3, zorder=1)
    
    # Corpus documents as background
    plt.scatter(corpus_points[:, 0], corpus_points[:, 1], 
               c='lightgray', s=30, alpha=0.3, zorder=0)
    
    # Retrieved documents with color gradient based on distance
    norm_distances = (distances - distances.min()) / (distances.max() - distances.min())
    scatter = plt.scatter(doc_points[:, 0], doc_points[:, 1],
                         c=norm_distances, cmap='viridis', 
                         s=100, alpha=0.8, zorder=3)
    
    # Add labels for retrieved documents with distances
    for i, (point, dist) in enumerate(zip(doc_points, distances)):
        plt.annotate(f'd{i+1}', (point[0], point[1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
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
