"""Tests for the RAG knowledge base module."""
import os
from src.rag.knowledge_base import load_articles, KB_DIR


class TestKnowledgeBase:
    """Test knowledge base loading and processing."""

    def test_load_articles_returns_documents(self):
        docs = load_articles()
        assert len(docs) >= 50

    def test_articles_have_content(self):
        docs = load_articles()
        for doc in docs:
            assert len(doc.page_content) > 0

    def test_articles_have_metadata(self):
        docs = load_articles()
        for doc in docs:
            assert "title" in doc.metadata
            assert "category" in doc.metadata
            assert "source" in doc.metadata

    def test_articles_have_valid_categories(self):
        valid_categories = {
            "basics", "strategy", "planning", "tax", "analysis",
            "alternatives", "advanced", "reference", "general", "retirement"
        }
        docs = load_articles()
        for doc in docs:
            assert doc.metadata["category"] in valid_categories, \
                f"Invalid category '{doc.metadata['category']}' in {doc.metadata['source']}"

    def test_knowledge_base_directory_exists(self):
        assert os.path.isdir(KB_DIR)

    def test_markdown_files_exist(self):
        import glob
        files = glob.glob(os.path.join(KB_DIR, "*.md"))
        assert len(files) >= 50

    def test_articles_have_frontmatter(self):
        docs = load_articles()
        for doc in docs:
            assert "title:" in doc.page_content or doc.metadata["title"] != ""

    def test_no_empty_titles(self):
        docs = load_articles()
        for doc in docs:
            assert doc.metadata["title"].strip() != ""
