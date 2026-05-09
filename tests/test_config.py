"""Tests for the configuration module."""
import os
import yaml
from src.core.config import load_config, config


class TestConfig:
    """Test configuration loading."""

    def test_config_loads(self):
        assert config is not None
        assert isinstance(config, dict)

    def test_config_has_llm_section(self):
        assert "llm" in config
        assert "model" in config["llm"]
        assert "temperature" in config["llm"]

    def test_config_has_embedding_section(self):
        assert "embedding" in config
        assert "model" in config["embedding"]

    def test_config_has_rag_section(self):
        assert "rag" in config
        assert "chunk_size" in config["rag"]
        assert "chunk_overlap" in config["rag"]
        assert "top_k" in config["rag"]

    def test_config_has_app_section(self):
        assert "app" in config
        assert "name" in config["app"]
        assert "disclaimer" in config["app"]

    def test_config_values_types(self):
        assert isinstance(config["llm"]["temperature"], (int, float))
        assert isinstance(config["rag"]["chunk_size"], int)
        assert isinstance(config["rag"]["top_k"], int)
        assert isinstance(config["app"]["name"], str)

    def test_config_reasonable_values(self):
        assert 0 <= config["llm"]["temperature"] <= 1
        assert config["rag"]["chunk_size"] > 0
        assert config["rag"]["chunk_overlap"] >= 0
        assert config["rag"]["chunk_overlap"] < config["rag"]["chunk_size"]
        assert config["rag"]["top_k"] > 0

    def test_load_config_custom_path(self):
        # Create a temp config
        temp_path = "/tmp/test_config.yaml"
        test_data = {"llm": {"model": "test-model", "temperature": 0.5}}
        with open(temp_path, "w") as f:
            yaml.dump(test_data, f)

        result = load_config(temp_path)
        assert result["llm"]["model"] == "test-model"
        os.remove(temp_path)

    def test_app_name_is_finnie(self):
        assert config["app"]["name"] == "Finnie"
