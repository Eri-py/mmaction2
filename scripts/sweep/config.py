import yaml


def load_config(config_path):
    """Load and parse the model-sweep YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
