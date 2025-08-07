"""
Utility functions for the text-style-mimicry project.
"""

import yaml

def load_config(path="config/config.yml"):
    """
    Loads the YAML configuration file.

    Args:
        path (str): The path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)
