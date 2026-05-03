import yaml
import os


def load_config(path: str = None) -> dict:
    """Load configuration from YAML file."""
    if path is None:
        # Walk up from this file to find config.yaml at project root
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, "config.yaml")

    with open(path) as f:
        return yaml.safe_load(f)


# Load once, import everywhere
config = load_config()