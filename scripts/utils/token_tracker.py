#!/usr/bin/env python3
import logging
import tiktoken
from typing import Dict, Any, List, Tuple, Optional

class TokenTracker:
    """A class to handle token tracking for various LLM providers."""
    
    def __init__(self):
        """Initialize the token tracker."""
        self.encoders = {}
    
    def get_encoder(self, model_name: str) -> Any:
        """Get or create a tiktoken encoder for a given model."""
        if model_name not in self.encoders:
            try:
                # Try to get the right encoder for the model
                if "gpt-4" in model_name.lower():
                    self.encoders[model_name] = tiktoken.encoding_for_model("gpt-4")
                elif "gpt-3.5" in model_name.lower():
                    self.encoders[model_name] = tiktoken.encoding_for_model("gpt-3.5-turbo")
                elif "llama" in model_name.lower():
                    # Use cl100k_base for LLaMa models
                    self.encoders[model_name] = tiktoken.get_encoding("cl100k_base")
                elif "claude" in model_name.lower():
                    # Claude uses cl100k_base
                    self.encoders[model_name] = tiktoken.get_encoding("cl100k_base")
                elif "mistral" in model_name.lower():
                    # Mistral also uses cl100k_base
                    self.encoders[model_name] = tiktoken.get_encoding("cl100k_base")
                else:
                    # Default to cl100k_base for unknown models
                    self.encoders[model_name] = tiktoken.get_encoding("cl100k_base")
                    logging.info(f"Using default encoding for unknown model: {model_name}")
            except Exception as e:
                logging.warning(f"Error creating encoder for {model_name}: {e}")
                # Fall back to a default encoding
                self.encoders[model_name] = tiktoken.get_encoding("cl100k_base")
        
        return self.encoders[model_name]
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """Count tokens in a text string for a specific model."""
        encoder = self.get_encoder(model_name)
        return len(encoder.encode(text))
    
    def estimate_openai_tokens(self, system_prompt: str, user_input: str, model_name: str) -> int:
        """Estimate tokens for OpenAI-compatible API format."""
        # Format as OpenAI messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Count tokens according to OpenAI's format
        # Each message has a base token cost (4 tokens for metadata)
        # Plus the content tokens
        encoder = self.get_encoder(model_name)
        total_tokens = 0
        
        for message in messages:
            # Add message metadata tokens (role + other formatting)
            total_tokens += 4
            # Add content tokens
            total_tokens += len(encoder.encode(message["content"]))
        
        # Add final assistant message token
        total_tokens += 2
        
        return total_tokens
    
    def estimate_claude_tokens(self, system_prompt: str, user_input: str, model_name: str) -> int:
        """Estimate tokens for Claude API format."""
        # Claude format: system prompt + Human: user_input \n\nAssistant:
        formatted_input = f"{system_prompt}\n\nHuman: {user_input}\n\nAssistant:"
        return self.count_tokens(formatted_input, model_name)
    
    def extract_tokens_from_response(self, response: Any, provider: str, 
                                     system_prompt: str = "", user_input: str = "",
                                     response_text: str = "", model_name: str = "") -> Dict[str, int]:
        """
        Extract token counts from a response object or estimate if not available.
        
        Args:
            response: The response object from the LLM provider
            provider: The LLM provider name
            system_prompt: The system prompt used (for estimation)
            user_input: The user input used (for estimation)
            response_text: The response text (for estimation)
            model_name: The model name (for estimation)
            
        Returns:
            Dict with 'input_tokens' and 'output_tokens'
        """
        token_usage = {'input_tokens': 0, 'output_tokens': 0}
        
        # Handle the case when response is None or an error occurred
        if response is None:
            logging.warning(f"Response object is None, using estimation for {provider}")
            return self.get_token_usage_summary(
                provider, model_name, system_prompt, user_input, response_text
            )
        
        # Try to extract from response first
        try:
            if provider in ['openai', 'azureopenai', 'deepinfra', 'ollama']:
                if hasattr(response, 'usage') and response.usage is not None:
                    # Direct extraction from response usage
                    token_usage['input_tokens'] = getattr(response.usage, 'prompt_tokens', 0)
                    token_usage['output_tokens'] = getattr(response.usage, 'completion_tokens', 0)
                    
                    # If we have valid token counts, return them
                    if token_usage['input_tokens'] > 0 or token_usage['output_tokens'] > 0:
                        return token_usage
            
            elif provider == 'claude':
                if hasattr(response, 'usage'):
                    # Direct extraction from Claude response
                    token_usage['input_tokens'] = getattr(response.usage, 'input_tokens', 0)
                    token_usage['output_tokens'] = getattr(response.usage, 'output_tokens', 0)
                    
                    # If we have valid token counts, return them
                    if token_usage['input_tokens'] > 0 or token_usage['output_tokens'] > 0:
                        return token_usage
        except Exception as e:
            logging.warning(f"Error extracting tokens from response: {e}. Using estimation.")
        
        # If we couldn't extract tokens or they're all zeros, estimate them
        if system_prompt and user_input and model_name:
            return self.get_token_usage_summary(
                provider, model_name, system_prompt, user_input, response_text
            )
        
        return token_usage
    
    def get_token_usage_summary(self, provider: str, model_name: str, 
                               system_prompt: str, user_input: str, 
                               response_text: str) -> Dict[str, int]:
        """
        Get a complete token usage summary without an API response object.
        Useful for testing or when API doesn't return token counts.
        
        Args:
            provider: The LLM provider name
            model_name: The model name
            system_prompt: The system prompt used
            user_input: The user input used
            response_text: The response text
            
        Returns:
            Dict with 'input_tokens', 'output_tokens', and 'total_tokens'
        """
        if provider in ['openai', 'azureopenai', 'deepinfra']:
            input_tokens = self.estimate_openai_tokens(system_prompt, user_input, model_name)
        elif provider == 'claude':
            input_tokens = self.estimate_claude_tokens(system_prompt, user_input, model_name)
        else:
            # Generic estimation
            combined_input = f"{system_prompt}\n\n{user_input}"
            input_tokens = self.count_tokens(combined_input, model_name)
        
        output_tokens = self.count_tokens(response_text, model_name) if response_text else 0
        
        return {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens
        }