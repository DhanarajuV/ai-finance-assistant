from langchain_core.tools import tool
from src.rag.knowledge_base import load_vector_store
from src.core.config import config

# Load once when imported
_vector_store = load_vector_store()


@tool
def search_knowledge_base(query: str) -> str:
    """Search the financial education knowledge base for relevant information.

    Use this tool when the user asks about financial concepts like stocks, bonds,
    ETFs, retirement accounts, diversification, or investing basics.

    Args:
        query: The user's question or topic to search for.
    """
    results = _vector_store.similarity_search(query, k=config["rag"]["top_k"])

    if not results:
        return "No relevant articles found in the knowledge base."

    output = ""
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("title", "Unknown")
        output += f"\n[Source {i}: {source}]\n{doc.page_content}\n"

    return output
