from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from src.config.settings import GOOGLE_API_KEY, LLM_MODEL, EMBEDDING_MODEL

def init_llm():
    """Initialize the LLM model."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    )

def init_embeddings():
    """Initialize the embeddings model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_query"
    )
