#!/usr/bin/env python3
import os
import json
import logging
from typing import Dict, Any

def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    Assumes this module is in the utils/ directory under scripts/ under the project root.
    
    Returns:
        str: Absolute path to the project root directory
    """
    # Get the directory of the current module (utils/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to get to the project root (from utils/ to scripts/ to project root)
    return os.path.dirname(os.path.dirname(current_dir))

def load_config(config_path: str = None, default_config: Dict = None) -> Dict:
    """
    Load configuration from a JSON file. If file doesn't exist or path not provided,
    return the default config.
    
    Args:
        config_path: Path to the configuration JSON file
        default_config: Default configuration to use if file not found
        
    Returns:
        Dict: Configuration dictionary
    """
    if config_path and os.path.isfile(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logging.warning(f"Error loading configuration from {config_path}: {e}")
            logging.info("Using default configuration")
            return default_config or {}
    else:
        if config_path:
            logging.warning(f"Configuration file not found: {config_path}")
            logging.info("Using default configuration")
        return default_config or {}

def load_environment_variables() -> None:
    """
    This function is kept for backward compatibility but is no longer needed
    since we're using API keys directly from the configuration file.
    """
    logging.info("Using API keys from configuration file instead of environment variables")

def inject_environment_variables(config: Dict) -> Dict:
    """
    This function is kept for backward compatibility but now just returns the config
    since we're using API keys directly from the configuration file.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dict: The same configuration dictionary
    """
    logging.info("Using API keys from configuration file instead of environment variables")
    return config

# In scripts/utils/config_loader.py

def validate_api_keys(config: Dict) -> bool:
    """
    Validates that the API key for the active provider is available,
    checking environment variables first, then the config file.
    """
    model_settings = config.get('model_settings', {})
    provider = model_settings.get('active_provider')
    if not provider:
        logging.error("No 'active_provider' found in model_settings.")
        return False

    # Check for the key in the environment first. The orchestrator sets 'LLM_API_KEY'.
    if 'LLM_API_KEY' in os.environ and os.environ['LLM_API_KEY']:
        logging.info(f"API key found in 'LLM_API_KEY' environment variable for provider '{provider}'. Validation successful.")
        # We need to inject this key back into the config for the LLMProvider to use it.
        # This is a critical step.
        if provider in ['openai', 'claude', 'azureopenai']:
            config['model_settings'][provider]['api_key'] = os.environ['LLM_API_KEY']
        elif provider in ['deepinfra', 'ollama']:
            active_model_key = config['model_settings'][provider].get('active_model')
            if active_model_key:
                config['model_settings'][provider]['models'][active_model_key]['api_key'] = os.environ['LLM_API_KEY']
        return True

    # If not in env, check the config file as a fallback
    provider_settings = model_settings.get(provider, {})
    api_key_from_config = ""
    if provider in ['deepinfra', 'ollama']:
        active_model_key = provider_settings.get('active_model')
        if active_model_key and 'models' in provider_settings:
            model_config = provider_settings['models'].get(active_model_key, {})
            api_key_from_config = model_config.get('api_key')
    else:
        api_key_from_config = provider_settings.get('api_key')

    if api_key_from_config:
        logging.info(f"API key found in config file for provider '{provider}'.")
        return True

    # If no key is found anywhere
    logging.error(f"API key for active provider '{provider}' is MISSING.")
    logging.error("Please set the LLM_API_KEY environment variable or add the key to your config_llm.json.")
    return False