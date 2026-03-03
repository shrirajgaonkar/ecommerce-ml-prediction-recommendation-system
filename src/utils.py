import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    path = Path(config_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / config_path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
