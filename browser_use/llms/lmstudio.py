"""LMStudio integration for browser-use.

This module provides integration with LMStudio's local API server, allowing you to use
your own local language models with browser-use.

Example:
    ```python
    from browser_use import Browser
    from browser_use.llms import LMStudioLLM

    # Initialize LMStudio client
    llm = LMStudioLLM(
        base_url="http://localhost:1234/v1",  # Default LMStudio API endpoint
        temperature=0.7,
        max_tokens=512
    )

    # Initialize Browser with LMStudio
    browser = Browser(llm=llm)

    # Use it like any other model
    response = await browser.browse("https://example.com")
    ```

Make sure to:
1. Have LMStudio running locally
2. Load a language model in LMStudio
3. Enable the API server in LMStudio
"""

from typing import Any, Dict, List, Optional
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class LMStudioLLM(BaseLLM):
    """LMStudio LLM wrapper."""
    
    base_url: str = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
    model: str = os.getenv("LMSTUDIO_MODEL", "local-model")
    temperature: float = 0.7
    max_tokens: int = 512
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the LMStudio API."""
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        if stop is not None:
            data["stop"] = stop
            
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(
                f"LMStudio API returned error {response.status_code}: {response.text}"
            )
            
        return response.json()["choices"][0]["message"]["content"]
    
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "lmstudio"
        
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
