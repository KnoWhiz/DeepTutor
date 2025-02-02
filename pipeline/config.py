import json
from pathlib import Path

def load_config():
    """Load configuration from config.json"""
    config_path = Path(__file__).parent / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f) 