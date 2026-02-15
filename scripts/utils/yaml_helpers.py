"""
YAML handling utilities for Step 2.
"""

from pathlib import Path
from typing import Dict, Any
import logging

try:
    from ruamel.yaml import YAML
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.width = 120
    HAS_RUAMEL = True
except ImportError:
    import yaml as pyyaml
    HAS_RUAMEL = False

def save_yaml(data: Dict[str, Any], output_path: Path):
    """Save dictionary to YAML file with proper formatting."""
    logging.info(f"Saving YAML to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if HAS_RUAMEL:
            yaml.dump(data, f)
        else:
            pyyaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    logging.info("YAML saved successfully")

def save_yaml_with_comments(data: Dict[str, Any], output_path: Path):
    """Save YAML with helpful comments for missing fields."""
    
    # Add comments for tasks with missing commands
    if 'tasks' in data:
        for task in data['tasks']:
            if 'command' not in task.get('config', {}):
                if 'validation_warnings' not in task:
                    task['validation_warnings'] = []
                task['validation_warnings'].append(
                    "No command specified - uses image default ENTRYPOINT/CMD"
                )

def load_yaml(input_path: Path) -> Dict[str, Any]:
    """Load YAML file into dictionary."""
    logging.info(f"Loading YAML from {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        if HAS_RUAMEL:
            data = yaml.load(f)
        else:
            data = pyyaml.safe_load(f)
    
    return data