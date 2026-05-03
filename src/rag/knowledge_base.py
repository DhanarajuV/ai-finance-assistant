import os
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
from src.core.config import config

load_dotenv()

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base")
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")


def _get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=config["embedding"]["model"])


def load_articles() -> list[Document]:
    """Load all markdown articles from the knowledge base directory."""
    documents = []
    for filepath in glob.glob(os.path.join(KB_DIR, "*.md")):
        with open(filepath) as f:
            content = f.read()

        title = os.path.basename(filepath).replace("_", " ").replace(".md", "").title()
        if "title:" in content:
            for line in content.split("\n"):
                if line.strip().startswith("title:"):
                    title = line.split(":", 1)[1].strip()
                    break

        category = "general"
        if "category:" in content:
            for line in content.split("\n"):
                if line.strip().startswith("category:"):
                    category = line.split(":", 1)[1].strip()
                    break

        documents.append(Document(
            page_content=content,
            metadata={"source": os.path.basename(filepath), "title": title, "category": category},
        ))

    print(f"Loaded {len(documents)} articles")
    return documents


def build_vector_store() -> FAISS:
    """Build FAISS vector store from knowledge base articles."""
    documents = load_articles()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["rag"]["chunk_size"],
        chunk_overlap=config["rag"]["chunk_overlap"],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    vector_store = FAISS.from_documents(chunks, _get_embeddings())

    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"Vector store saved to {FAISS_INDEX_DIR}")
    return vector_store


def load_vector_store() -> FAISS:
    """Load existing vector store, or build if it doesn't exist."""
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(FAISS_INDEX_DIR, _get_embeddings(), allow_dangerous_deserialization=True)
    return build_vector_store()


if __name__ == "__main__":
    vs = build_vector_store()

    results = vs.similarity_search("What is an ETF?", k=config["rag"]["top_k"])
    print("\n--- Search results for 'What is an ETF?' ---")
    for doc in results:
        print(f"\nSource: {doc.metadata['title']}")
        print(f"Content: {doc.page_content[:150]}...")