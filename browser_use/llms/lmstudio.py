"""LMStudio integration for browser-use.

This module provides integration with LMStudio's local API server, allowing you to use
your own local language models with browser-use.

Example:
    ```python
    from browser_use import Browser, Agent
    from browser_use.llms import LMStudioLLM

    # Initialize LMStudio client
    llm = LMStudioLLM(
        base_url="http://localhost:1234/v1",  # Default LMStudio API endpoint
        temperature=0.7,
        max_tokens=512
    )

    # Initialize Browser and Agent
    browser = Browser()
    agent = Agent(task="Go to example.com", llm=llm, browser=browser)
    result = await agent.run()
    ```

Make sure to:
1. Have LMStudio running locally
2. Load a language model in LMStudio
3. Enable the API server in LMStudio
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Iterator, Mapping, Tuple
import os
import json
import requests
import logging
from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatGeneration, ChatResult

load_dotenv()

# Set debug level for this module
logging.getLogger(__name__).setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text.
    This is a rough estimate based on the number of words.
    A more accurate count would require the actual tokenizer.
    """
    # A rough estimate: 1 token ≈ 4 characters
    return len(text) // 4


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    current_tokens = estimate_tokens(text)
    if current_tokens <= max_tokens:
        return text

    # Truncate to approximately max_tokens by characters
    max_chars = max_tokens * 4
    return text[:max_chars] + "..."


class LMStudioLLM(BaseChatModel):
    """LMStudio chat model wrapper."""

    base_url: str = os.getenv("LMSTUDIO_API_URL", "http://localhost:1234/v1")
    model: str = os.getenv("LMSTUDIO_MODEL", "local-model")
    temperature: float = 0.7
    max_tokens: int = 512
    tool_calling_method: Optional[str] = None
    _output_schema: Optional[Any] = None
    context_length: int = int(os.getenv("LMSTUDIO_CONTEXT_LENGTH", "8192"))
    max_input_tokens: int = (
        context_length - max_tokens - 100
    )  # Leave some room for the response

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "lmstudio"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to prompt format."""
        logger.debug("Converting messages to prompt format")
        prompt = ""

        # If we have an output schema, add it as a system message
        if self._output_schema is not None:
            schema = self._output_schema.model_json_schema()
            required_fields = schema.get("required", [])
            properties = schema.get("properties", {})

            prompt += "You are a JSON-only assistant. You must respond with valid JSON that matches this schema:\n\n"
            prompt += "{\n"
            for field in required_fields:
                field_type = properties[field].get("type", "any")
                field_desc = properties[field].get("description", "")
                prompt += f'  "{field}": {field_type}, // {field_desc}\n'
            prompt += "}\n\n"
            prompt += "Example response:\n"
            prompt += "{\n"
            prompt += '  "current_state": {\n'
            prompt += '    "evaluation_previous_goal": "Success - The page loaded correctly",\n'
            prompt += '    "memory": "We are on the Google homepage",\n'
            prompt += '    "next_goal": "We need to search for something"\n'
            prompt += "  },\n"
            prompt += '  "action": [\n'
            prompt += "    {\n"
            prompt += '      "action_name": "go_to_url",\n'
            prompt += '      "parameters": {"url": "https://www.example.com"}\n'
            prompt += "    }\n"
            prompt += "  ]\n"
            prompt += "}\n\n"
            prompt += "IMPORTANT: Your response must start with '{' and end with '}' and be valid JSON!\n"
            prompt += "IMPORTANT: The field names are case-sensitive! Use exactly: evaluation_previous_goal, memory, next_goal\n"
            prompt += (
                "IMPORTANT: The 'current_state' object must include ALL fields!\n\n"
            )

        # Only use the last message to keep the prompt short
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, SystemMessage):
                prompt += f"System: {last_message.content}\n"
            elif isinstance(last_message, HumanMessage):
                prompt += f"Human: {truncate_text(str(last_message.content), self.max_input_tokens // 2)}\n"
            elif isinstance(last_message, AIMessage):
                prompt += f"Assistant: {truncate_text(str(last_message.content), self.max_input_tokens // 2)}\n"
            else:
                logger.warning(f"Unknown message type: {type(last_message)}")
                prompt += f"Human: {truncate_text(str(last_message.content), self.max_input_tokens // 2)}\n"

        prompt += "Assistant (remember to respond with valid JSON): "

        # Log estimated token count
        estimated_tokens = estimate_tokens(prompt)
        logger.info(f"Estimated input tokens: {estimated_tokens}")
        logger.info(f"Available context length: {self.context_length}")
        logger.info(
            f"Remaining tokens for completion: {self.context_length - estimated_tokens}"
        )

        logger.debug(f"Converted prompt: {prompt}")
        return prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion synchronously."""
        headers = {"Content-Type": "application/json"}

        prompt = self._convert_messages_to_prompt(messages)
        logger.debug(f"Sending request to LMStudio API at {self.base_url}")
        logger.debug(f"Prompt: {prompt}")

        data = {
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        if stop is not None:
            data["stop"] = stop

        try:
            response = requests.post(
                f"{self.base_url}/completions", headers=headers, json=data
            )

            logger.debug(f"Response status code: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"Error response: {response.text}")
                raise ValueError(
                    f"LMStudio API returned error {response.status_code}: {response.text}"
                )

            response_json = response.json()
            logger.debug(f"Response JSON: {response_json}")
            response_text = response_json["choices"][0]["text"].strip()
            logger.debug(f"Response text: {response_text}")

            # Try to parse as JSON
            try:
                # Make sure response starts with { and ends with }
                if not response_text.startswith("{") or not response_text.endswith("}"):
                    logger.warning("Response text is not a valid JSON object")
                    # Try to find JSON object in the response
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    if start >= 0 and end > start:
                        response_text = response_text[start:end]
                        logger.info(f"Extracted JSON: {response_text}")
                    else:
                        raise json.JSONDecodeError(
                            "No JSON object found", response_text, 0
                        )

                # Clean up the response text
                response_text = response_text.strip()
                response_text = response_text.replace("\n", "")
                response_text = response_text.replace("\r", "")
                response_text = response_text.replace("\t", "")

                parsed = json.loads(response_text)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Parsed response is not a dictionary: {parsed}")

                # Fix case sensitivity issues
                if "current_state" in parsed and isinstance(
                    parsed["current_state"], dict
                ):
                    current_state = parsed["current_state"]
                    if "evaluation_previous_Goal" in current_state:
                        current_state["evaluation_previous_goal"] = current_state.pop(
                            "evaluation_previous_Goal"
                        )
                    if "next_Goal" in current_state:
                        current_state["next_goal"] = current_state.pop("next_Goal")

                # Validate required fields
                if self._output_schema is not None:
                    schema = self._output_schema.model_json_schema()
                    required_fields = schema.get("required", [])
                    for field in required_fields:
                        if field not in parsed:
                            raise ValueError(
                                f"Required field '{field}' missing from response"
                            )

                additional_kwargs = {"parsed": parsed}
                message = AIMessage(
                    content=response_text, additional_kwargs=additional_kwargs
                )
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse response as JSON: {str(e)}")
                message = AIMessage(content=response_text)

            generation = ChatGeneration(
                message=message,
                generation_info={
                    "finish_reason": response_json["choices"][0].get("finish_reason")
                },
            )
            return ChatResult(generations=[generation])

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            raise ValueError(f"Failed to connect to LMStudio API: {str(e)}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion asynchronously."""
        return self._generate(messages, stop, run_manager, **kwargs)

    def with_structured_output(self, output_schema: Any, **kwargs: Any) -> LMStudioLLM:
        """Add structured output to the model."""
        # Create a new instance with the same parameters
        new_instance = self.__class__(
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        new_instance._output_schema = output_schema
        return new_instance
