"""
LLM Client for SM-UMT.
Abstraction layer for Gemini API using the new google-genai package.
"""

import os
import time
from typing import List, Optional
from google import genai
from google.genai import types


class GeminiClient:
    """
    Client for Google's Gemini API using the new google-genai package.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.3,
        max_tokens: int = 256
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model_name: Model to use (gemini-2.0-flash, gemini-1.5-flash, etc.)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Initialize client with API key
        self.client = genai.Client(api_key=self.api_key)
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Rate limiting - 2 seconds between requests to stay within free tier limits
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests for free tier
    
    def _rate_limit(self):
        """Apply rate limiting to avoid API throttling."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
        
        Returns:
            Generated text response
        """
        self._rate_limit()
        
        try:
            # Create generation config
            config = types.GenerateContentConfig(
                temperature=temperature or self.temperature,
                max_output_tokens=max_tokens or self.max_tokens,
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # Extract text from response
            if response.text:
                return response.text.strip()
            return ""
            
        except Exception as e:
            print(f"Error generating response: {e}")
            # Retry once after a delay
            time.sleep(1)
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                return response.text.strip() if response.text else ""
            except Exception as e2:
                print(f"Retry failed: {e2}")
                return ""
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of input prompts
            temperature: Override default temperature
            max_tokens: Override default max tokens
        
        Returns:
            List of generated text responses
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, temperature, max_tokens)
            responses.append(response)
        return responses
    
    def get_stats(self) -> dict:
        """Get client statistics."""
        return {
            "model": self.model_name,
            "request_count": self.request_count,
        }


class LLMClient:
    """
    Unified LLM client interface.
    Currently supports Gemini, can be extended for other providers.
    """
    
    def __init__(
        self,
        provider: str = "gemini",
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            provider: LLM provider ("gemini")
            **kwargs: Provider-specific arguments
        """
        self.provider = provider
        
        if provider == "gemini":
            self.client = GeminiClient(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response."""
        return self.client.generate(prompt, **kwargs)
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        return self.client.generate_batch(prompts, **kwargs)
    
    def get_stats(self) -> dict:
        """Get client statistics."""
        return self.client.get_stats()
